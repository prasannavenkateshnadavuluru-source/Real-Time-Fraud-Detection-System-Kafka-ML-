import time
import random
import datetime
import threading
import math
from typing import Dict, Any, List
import logging

from src.broker import Producer

logger = logging.getLogger("data_generator")

LOCATIONS = [
    {"city": "New York", "lat": 40.7128, "lon": -74.0060},
    {"city": "Los Angeles", "lat": 34.0522, "lon": -118.2437},
    {"city": "Chicago", "lat": 41.8781, "lon": -87.6298},
    {"city": "Houston", "lat": 29.7604, "lon": -95.3698},
    {"city": "Phoenix", "lat": 33.4484, "lon": -112.0740},
    {"city": "Miami", "lat": 25.7617, "lon": -80.1918},
    {"city": "San Francisco", "lat": 37.7749, "lon": -122.4194},
    {"city": "Seattle", "lat": 47.6062, "lon": -122.3321}
]

CATEGORIES = ["grocery", "retail", "food", "entertainment", "travel", "utilities", "online_shopping"]

generator_settings = {
    "inject_fraud_burst": False,
    "is_running": False,
    "sleep_interval_sec": 1.2
}

class TransactionGenerator:
    def __init__(self, num_users: int = 50):
        self.producer = Producer(client_id="transaction-generator-producer")
        self.users = []
        self._lock = threading.Lock()
        self._thread = None
        
        # Pre-generate user base
        for i in range(num_users):
            u_id = f"user_{2000 + i}"
            base_loc = random.choice(LOCATIONS)
            avg_amount = random.uniform(20.0, 150.0)
            self.users.append({
                "user_id": u_id,
                "lat": base_loc["lat"],
                "lon": base_loc["lon"],
                "avg_amount": avg_amount,
                "last_lat": base_loc["lat"],
                "last_lon": base_loc["lon"],
                "last_tx_time": None
            })

    def start(self):
        with self._lock:
            if generator_settings["is_running"]:
                return
            generator_settings["is_running"] = True
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            logger.info("Transaction streaming generator thread started.")

    def stop(self):
        with self._lock:
            generator_settings["is_running"] = False
        if self._thread:
            self._thread.join(timeout=2.0)
            logger.info("Transaction streaming generator thread stopped.")

    def _run_loop(self):
        tx_counter = 0
        while generator_settings["is_running"]:
            try:
                # Determine sleep interval
                sleep_time = generator_settings["sleep_interval_sec"]
                
                # Check for fraud injection burst
                if generator_settings["inject_fraud_burst"]:
                    # Create multiple suspicious transactions quickly
                    logger.info("INJECTING FRAUD BURST TRANSACTIONS...")
                    for _ in range(3):
                        self._generate_transaction(is_fraudulent=True, tx_id=f"tx_gen_{tx_counter}")
                        tx_counter += 1
                        time.sleep(0.1)
                    # Turn off burst flag after executing
                    generator_settings["inject_fraud_burst"] = False
                else:
                    # 2.5% natural background fraud rate in simulator
                    is_natural_fraud = random.random() < 0.025
                    self._generate_transaction(is_fraudulent=is_natural_fraud, tx_id=f"tx_gen_{tx_counter}")
                    tx_counter += 1
                
                time.sleep(random.uniform(sleep_time * 0.8, sleep_time * 1.2))
            except Exception as e:
                logger.error(f"Error in data generator thread: {e}", exc_info=True)
                time.sleep(1)

    def _generate_transaction(self, is_fraudulent: bool, tx_id: str):
        # Pick a random user
        user = random.choice(self.users)
        u_id = user["user_id"]
        
        now = datetime.datetime.utcnow().isoformat()
        
        if is_fraudulent:
            fraud_type = random.choice(["amount_spike", "velocity_spike", "geolocation_jump"])
            
            if fraud_type == "amount_spike":
                # High-value amount spike
                amount = round(user["avg_amount"] * random.uniform(15.0, 35.0), 2)
                lat = user["last_lat"] + random.uniform(-0.01, 0.01)
                lon = user["last_lon"] + random.uniform(-0.01, 0.01)
                category = "travel" if amount > 500 else "online_shopping"
                
            elif fraud_type == "velocity_spike":
                # Instant transaction (velocity count spike)
                amount = round(user["avg_amount"] * random.uniform(1.5, 3.0), 2)
                lat = user["last_lat"] + random.uniform(-0.005, 0.005)
                lon = user["last_lon"] + random.uniform(-0.005, 0.005)
                category = "retail"
                
            elif fraud_type == "geolocation_jump":
                # Location jump: select a city far away from user's current city
                far_loc = random.choice([loc for loc in LOCATIONS if abs(loc["lat"] - user["last_lat"]) > 5.0])
                lat = far_loc["lat"] + random.uniform(-0.1, 0.1)
                lon = far_loc["lon"] + random.uniform(-0.1, 0.1)
                amount = round(random.uniform(30.0, 300.0), 2)
                category = "food"
                
            is_fraud_labeled = 1
        else:
            # Normal transaction behavior
            amount = float(random.lognormvariate(mu=math.log(user["avg_amount"]), sigma=0.35))
            amount = max(1.0, round(amount, 2))
            
            # Tiny movement around last known location
            lat = user["last_lat"] + random.uniform(-0.015, 0.015)
            lon = user["last_lon"] + random.uniform(-0.015, 0.015)
            category = random.choice(CATEGORIES)
            is_fraud_labeled = 0

        # Update user's last known state
        user["last_lat"] = lat
        user["last_lon"] = lon
        user["last_tx_time"] = now
        
        # Construct transaction payload
        tx_payload = {
            "transaction_id": tx_id,
            "user_id": u_id,
            "amount": amount,
            "timestamp": now,
            "lat": lat,
            "lon": lon,
            "merchant_category": category,
            "device_id": f"device_{u_id[-4:]}",
            "ip_address": f"192.168.1.{random.randint(2, 254)}",
            "is_fraud_labeled": is_fraud_labeled
        }
        
        # Produce message to topic
        self.producer.send(topic="transactions", value=tx_payload, key=u_id)
        logger.info(f"Produced transaction {tx_id} for user {u_id} (Fraud={is_fraud_labeled}, Amt=${amount})")
