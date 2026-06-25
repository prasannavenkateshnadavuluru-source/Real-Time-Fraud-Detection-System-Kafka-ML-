import os
import asyncio
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import threading
import time

from src.db import (
    init_db, get_system_metrics, get_recent_transactions, 
    get_recent_alerts, get_dlq_events, clear_db
)
from src.pipeline import FraudDetectionPipeline, pipeline_settings
from src.data_generator import TransactionGenerator, generator_settings
from train import train_and_save_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fraud_app")

app = FastAPI(title="Real-Time Fraud Detection Dashboard")

# Paths setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(BASE_DIR, "src", "web", "static")
templates_dir = os.path.join(BASE_DIR, "src", "web", "templates")

# Ensure static and templates exist
os.makedirs(static_dir, exist_ok=True)
os.makedirs(templates_dir, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

# Pipeline and Generator instances
pipeline = FraudDetectionPipeline()
generator = TransactionGenerator()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self._lock = threading.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        with self._lock:
            self.active_connections.append(websocket)
        logger.info(f"New client connected. Active connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Active connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        # Gather active connections thread-safely
        with self._lock:
            connections = list(self.active_connections)
        
        if not connections:
            return
            
        for connection in connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                # Socket might have closed
                self.disconnect(connection)

manager = ConnectionManager()

# Background broadcast task
async def broadcast_metrics_loop():
    logger.info("Started real-time WebSockets metrics broadcast loop.")
    while True:
        try:
            if manager.active_connections:
                metrics = get_system_metrics()
                transactions = get_recent_transactions(limit=25)
                alerts = get_recent_alerts(limit=15)
                dlq = get_dlq_events()
                
                # Combine payload
                payload = {
                    "metrics": metrics,
                    "transactions": transactions,
                    "alerts": alerts,
                    "dlq": dlq,
                    "settings": {
                        "simulate_db_failure": pipeline_settings["simulate_db_failure"],
                        "generator_speed": generator_settings["sleep_interval_sec"],
                        "models_ready": pipeline.model_manager.is_ready()
                    }
                }
                await manager.broadcast(payload)
        except Exception as e:
            logger.error(f"Error in metrics broadcast: {e}", exc_info=True)
            
        await asyncio.sleep(1.0) # Broadcast every 1s

@app.on_event("startup")
def startup_event():
    # Initialize SQL database tables
    init_db()
    
    # Try to load models. If they don't exist, we can still start (using fallback rules)
    # The user can also trigger model training from the UI.
    pipeline.model_manager.load_models()
    
    # Start consumer pipeline
    pipeline.start()
    
    # Start streaming transaction generator
    generator.start()
    
    # Run the WebSockets broadcast task in the event loop
    asyncio.create_task(broadcast_metrics_loop())

@app.on_event("shutdown")
def shutdown_event():
    generator.stop()
    pipeline.stop()

# --- HTTP ENDPOINTS ---

@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    return templates.TemplateResponse(request, "index.html")

@app.get("/api/metrics")
def api_metrics():
    return get_system_metrics()

@app.get("/api/transactions")
def api_transactions(limit: int = 50):
    return get_recent_transactions(limit)

@app.get("/api/alerts")
def api_alerts(limit: int = 20):
    return get_recent_alerts(limit)

@app.get("/api/dlq")
def api_dlq():
    return get_dlq_events()

# --- CONTROLS ---

@app.post("/api/control/fraud_burst")
def trigger_fraud_burst():
    generator_settings["inject_fraud_burst"] = True
    return {"status": "success", "message": "Fraud burst scheduled."}

@app.post("/api/control/toggle_fault")
def toggle_db_fault():
    pipeline_settings["simulate_db_failure"] = not pipeline_settings["simulate_db_failure"]
    status_str = "ENABLED" if pipeline_settings["simulate_db_failure"] else "DISABLED"
    logger.info(f"Database fault simulation toggled to: {status_str}")
    return {"status": "success", "simulate_db_failure": pipeline_settings["simulate_db_failure"]}

class SpeedSettings(BaseModel):
    sleep_interval_sec: float

@app.post("/api/control/speed")
def update_generator_speed(speed: SpeedSettings):
    generator_settings["sleep_interval_sec"] = max(0.1, min(10.0, speed.sleep_interval_sec))
    logger.info(f"Updated generator sleep interval to: {generator_settings['sleep_interval_sec']}s")
    return {"status": "success", "sleep_interval_sec": generator_settings["sleep_interval_sec"]}

@app.post("/api/dlq/reprocess/{tx_id}")
def reprocess_dlq_transaction(tx_id: str):
    success = pipeline.reprocess_dlq_event(tx_id)
    if success:
        return {"status": "success", "message": f"Successfully reprocessed and resolved DLQ event {tx_id}."}
    else:
        return {"status": "error", "message": f"Failed to reprocess DLQ event {tx_id}."}

@app.post("/api/control/retrain")
def retrain_models():
    def retrain_worker():
        logger.info("Background model retraining initiated...")
        try:
            # Temporarily pause pipeline and generator to prevent DB locking issues
            generator.stop()
            time.sleep(1)
            
            train_and_save_models()
            
            # Reload models in the pipeline's manager
            pipeline.model_manager.load_models()
            
            # Restart
            pipeline.start()
            generator.start()
            logger.info("Background model retraining completed successfully.")
        except Exception as e:
            logger.error(f"Error during background model retraining: {e}", exc_info=True)
            # Make sure they are restarted
            pipeline.start()
            generator.start()
            
    threading.Thread(target=retrain_worker, daemon=True).start()
    return {"status": "success", "message": "Model retraining started in the background."}

@app.post("/api/control/clear_db")
def reset_system():
    clear_db()
    # Reset offsets in consumer to read from end (to simulate fresh start)
    pipeline.consumer.seek_to_end()
    logger.info("System database cleared. Consumer seeked to end.")
    return {"status": "success", "message": "System data cleared."}

# --- WEBSOCKET ENDPOINT ---

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # We don't expect messages from client for now, just keep connection open
            # Receive text raises ConnectionClosed when client disconnects
            _ = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
