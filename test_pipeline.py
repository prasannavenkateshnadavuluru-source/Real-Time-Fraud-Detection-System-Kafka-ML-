import unittest
import os
import sys
import tempfile
import json
import time

# Ensure src path is accessible
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.broker import MockKafkaBroker, Producer, Consumer
from src.features import compute_features, to_feature_vector
from src.db import init_db, save_transaction, get_user_history, get_db_connection
import src.db as db

class TestFraudPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Override DB path for testing to run in memory or temporary file
        cls.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        cls.temp_db_path = cls.temp_db.name
        cls.temp_db.close()
        
        # Patch db.DB_PATH
        db.DB_PATH = cls.temp_db_path
        init_db()

    @classmethod
    def tearDownClass(cls):
        # Clean up temporary database
        try:
            os.unlink(cls.temp_db_path)
        except Exception:
            pass

    def setUp(self):
        # Clear database and broker between tests
        conn = get_db_connection()
        conn.execute("DELETE FROM transactions")
        conn.execute("DELETE FROM alerts")
        conn.execute("DELETE FROM dlq_events")
        conn.commit()
        conn.close()
        
        # Clear mock broker topics
        MockKafkaBroker().clear()

    def test_mock_kafka_broker_pub_sub(self):
        broker = MockKafkaBroker()
        producer = Producer()
        consumer = Consumer(topic="test_topic", auto_offset_reset="earliest")
        
        test_payload = {"id": "1", "val": "hello"}
        producer.send(topic="test_topic", value=test_payload, key="key_1")
        
        # Poll message
        msgs = consumer.poll(timeout_ms=50, max_records=1)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].value, test_payload)
        self.assertEqual(msgs[0].key, "key_1")
        self.assertEqual(msgs[0].offset, 0)
        
        # Poll again (should be empty since offset advanced)
        msgs = consumer.poll(timeout_ms=50, max_records=1)
        self.assertEqual(len(msgs), 0)

    def test_feature_engineering_calculation(self):
        # Setup mock user transaction history
        tx_current = {
            "transaction_id": "tx_current",
            "user_id": "user_abc",
            "amount": 250.0,
            "timestamp": "2026-06-11T12:00:00",
            "lat": 40.7128,  # NY
            "lon": -74.0060
        }
        
        # History has 1 recent transaction 2 minutes ago in NY
        history = [
            {
                "transaction_id": "tx_prev",
                "user_id": "user_abc",
                "amount": 50.0,
                "timestamp": "2026-06-11T11:58:00",
                "lat": 40.7100,
                "lon": -74.0050,
                "status": "PROCESSED"
            }
        ]
        
        features = compute_features(tx_current, history)
        
        self.assertEqual(features["velocity_10m"], 1)
        self.assertEqual(features["velocity_1h"], 1)
        self.assertEqual(features["amount"], 250.0)
        # Ratio: 250.0 / 50.0 = 5.0
        self.assertAlmostEqual(features["amount_ratio_1h"], 5.0, places=2)
        # Time delta: 12:00:00 - 11:58:00 = 120 seconds
        self.assertEqual(features["time_delta_seconds"], 120.0)
        # Check distance is calculated (>0 km but small)
        self.assertTrue(0.0 < features["distance_delta_km"] < 10.0)

    def test_database_persistence_and_aggregations(self):
        tx1 = {
            "transaction_id": "tx1",
            "user_id": "user_1",
            "amount": 10.0,
            "timestamp": "2026-06-11T10:00:00",
            "lat": 40.0,
            "lon": -74.0,
            "merchant_category": "retail",
            "device_id": "d1",
            "ip_address": "127.0.0.1",
            "is_fraud_labeled": 0
        }
        save_transaction(tx1, status="PROCESSED")
        
        history = get_user_history("user_1", before_timestamp="2026-06-11T11:00:00")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["amount"], 10.0)
        
        # Future timestamp lookup shouldn't return history before it
        history_before = get_user_history("user_1", before_timestamp="2026-06-11T09:00:00")
        self.assertEqual(len(history_before), 0)

if __name__ == "__main__":
    unittest.main()
