import time
import json
import logging
import threading
from typing import Dict, Any, List, Optional

from src.broker import Consumer, Producer, Message
from src.db import save_transaction, save_alert, save_dlq, get_user_history, resolve_dlq_event
from src.features import compute_features
from src.models import ModelManager

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fraud_pipeline")

# Global state for simulating failures (toggled by the dashboard)
pipeline_settings = {
    "simulate_db_failure": False,
    "max_retries": 3,
    "backoff_multiplier": 0.1,  # Shortened for responsive simulation (0.1s, 0.2s, 0.4s)
    "is_running": False
}

class PipelineError(Exception):
    pass

class DatabaseConnectionError(PipelineError):
    pass

class FraudDetectionPipeline:
    def __init__(self):
        self.consumer = Consumer(topic="transactions", group_id="fraud-detection-group")
        self.dlq_producer = Producer(client_id="fraud-dlq-producer")
        self.model_manager = ModelManager()
        self._lock = threading.Lock()
        self._thread = None

    def start(self):
        with self._lock:
            if pipeline_settings["is_running"]:
                return
            pipeline_settings["is_running"] = True
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            logger.info("Fraud Detection Pipeline consumer thread started.")

    def stop(self):
        with self._lock:
            pipeline_settings["is_running"] = False
        if self._thread:
            self._thread.join(timeout=2.0)
            logger.info("Fraud Detection Pipeline consumer thread stopped.")

    def _run_loop(self):
        while pipeline_settings["is_running"]:
            try:
                # Poll for transaction messages
                msgs = self.consumer.poll(timeout_ms=200, max_records=1)
                if not msgs:
                    continue
                
                for msg in msgs:
                    tx_payload = msg.value
                    self._process_with_retry(tx_payload)
            except Exception as e:
                logger.error(f"Unexpected error in pipeline consumer loop: {e}", exc_info=True)
                time.sleep(1)

    def _process_with_retry(self, tx: Dict[str, Any]) -> bool:
        """
        Attempts to process a transaction with fault tolerance: retries & DLQ routing.
        """
        tx_id = tx["transaction_id"]
        retry_count = 0
        max_retries = pipeline_settings["max_retries"]
        
        while retry_count <= max_retries:
            try:
                # 1. Simulate Flaky DB/Network Connection if active
                if pipeline_settings["simulate_db_failure"]:
                    # Randomly fail 70% of the time during simulated outage to show retries
                    import random
                    if random.random() < 0.7:
                        # Log that we are retrying in DB
                        if retry_count > 0:
                            save_transaction(tx, status="RETRYING", retry_count=retry_count)
                        raise DatabaseConnectionError(
                            f"Database connection timed out during persistence of transaction {tx_id}."
                        )
                
                # 2. Extract transaction history for feature engineering
                user_id = tx["user_id"]
                history = get_user_history(user_id, tx["timestamp"])
                
                # 3. Compute Features
                features = compute_features(tx, history)
                
                # 4. Score Transaction using dual models
                prediction = self.model_manager.predict(features)
                
                # 5. Persist the outcome (successful branch)
                save_transaction(
                    tx=tx,
                    is_fraud_predicted=prediction["is_fraud"],
                    xgb_score=prediction["xgb_prob"],
                    iforest_score=prediction["iforest_score"],
                    status="PROCESSED",
                    retry_count=retry_count
                )
                
                # 6. Raise Alert if classified as fraud
                if prediction["is_fraud"] or prediction["alert_level"] in ("MEDIUM", "HIGH"):
                    details = (
                        f"Fraud detected! XGBoost score: {prediction['xgb_prob']:.2f}, "
                        f"Isolation Forest anomaly: {prediction['iforest_score']:.2f}. "
                        f"Features: Velocity(10m)={features['velocity_10m']}, "
                        f"Ratio={features['amount_ratio_1h']:.1f}, "
                        f"Dist={features['distance_delta_km']:.1f}km"
                    )
                    save_alert(tx_id, prediction["alert_level"], details)
                    logger.warning(f"ALERT [{prediction['alert_level']}]: Transaction {tx_id} flagged as fraud. {details}")
                
                logger.info(f"Successfully processed transaction {tx_id} (Attempt {retry_count + 1})")
                return True
                
            except DatabaseConnectionError as db_err:
                retry_count += 1
                logger.warning(f"Database error processing transaction {tx_id} (Attempt {retry_count}/{max_retries + 1}): {db_err}")
                
                if retry_count <= max_retries:
                    # Exponential Backoff
                    sleep_time = (2 ** retry_count) * pipeline_settings["backoff_multiplier"]
                    logger.info(f"Retrying transaction {tx_id} in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                else:
                    # Retry limit reached, route to DLQ
                    self._route_to_dlq(tx, str(db_err))
                    return False
            except Exception as e:
                # Any other unexpected validation error goes directly to DLQ
                logger.error(f"Fatal error processing transaction {tx_id}: {e}", exc_info=True)
                self._route_to_dlq(tx, f"Fatal processing error: {str(e)}")
                return False

    def _route_to_dlq(self, tx: Dict[str, Any], error_reason: str):
        tx_id = tx["transaction_id"]
        logger.error(f"DLQ WARNING: Routing transaction {tx_id} to Dead-Letter Queue. Reason: {error_reason}")
        
        # Save DLQ state to database for dashboard visibility
        save_dlq(tx_id, json.dumps(tx), error_reason)
        
        # Publish payload to DLQ topic in Mock Broker
        self.dlq_producer.send(topic="dlq_transactions", value=tx, key=tx_id)
        
        # Save transaction status as failed in database
        save_transaction(tx, status="DLQ", retry_count=pipeline_settings["max_retries"])

    def reprocess_dlq_event(self, tx_id: str) -> bool:
        """
        Manually triggers reprocessing of an event in the DLQ.
        This represents operational DLQ intervention.
        """
        from src.db import get_db_connection
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT raw_payload FROM dlq_events WHERE transaction_id = ? AND status = 'PENDING'", (tx_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            logger.error(f"Reprocess failed: Transaction {tx_id} not found in pending DLQ events.")
            return False
            
        tx_payload = json.loads(row["raw_payload"])
        logger.info(f"Manually reprocessing DLQ transaction: {tx_id}")
        
        # Temporarily bypass DB failure flag for this specific manual operation to ensure it succeeds
        original_db_fail_flag = pipeline_settings["simulate_db_failure"]
        pipeline_settings["simulate_db_failure"] = False
        
        success = False
        try:
            # Process the transaction
            success = self._process_with_retry(tx_payload)
            if success:
                resolve_dlq_event(tx_id)
                logger.info(f"DLQ transaction {tx_id} successfully reprocessed and resolved.")
        finally:
            pipeline_settings["simulate_db_failure"] = original_db_fail_flag
            
        return success
