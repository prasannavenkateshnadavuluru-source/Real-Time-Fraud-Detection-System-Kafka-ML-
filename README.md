# Real-Time-Fraud-Detection-System-Kafka-ML-
# Built a real-time fraud detection pipeline using Apache Kafka and Python for high-volume transaction processing,Developed ML models using XGBoost and Isolation Forest for anomaly detection and fraud prediction,Implemented real-time alerting, DLQs, retry mechanisms, and monitoring for reliable event processing. 
KAFKA_BROKER = "localhost:9092"
TRANSACTION_TOPIC = "transactions"
DLQ_TOPIC = "transactions_dlq"
from kafka import KafkaProducer
from config.kafka_config import *
import json
import random
import time

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda x: json.dumps(x).encode("utf-8")
)

while True:

    transaction = {
        "transaction_id": random.randint(100000,999999),
        "amount": random.randint(50,10000),
        "merchant_score": random.randint(1,100),
        "hour": random.randint(0,23)
    }
    producer.send(TRANSACTION_TOPIC, transaction)
    print("Transaction Sent:", transaction)
    time.sleep(2)
    import pandas as pd
import joblib

from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest

data = pd.read_csv("../data/fraud_data.csv")

X = data[['amount','merchant_score','hour']]
y = data['fraud']

xgb = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5
)
xgb.fit(X,y)
joblib.dump(xgb,"xgb_model.pkl")
iso = IsolationForest(
    contamination=0.03,
    random_state=42
)
iso.fit(X)
joblib.dump(iso,"iso_model.pkl")
print("Models Trained Successfully")
from kafka import KafkaConsumer, KafkaProducer
from config.kafka_config import *

from alerts.alert_service import send_alert

import json
import joblib

consumer = KafkaConsumer(
    TRANSACTION_TOPIC,
    bootstrap_servers=KAFKA_BROKER,
    value_deserializer=lambda x: json.loads(x.decode("utf-8"))
)

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda x: json.dumps(x).encode("utf-8")
)

xgb_model = joblib.load("../models/xgb_model.pkl")
iso_model = joblib.load("../models/iso_model.pkl")

print("Fraud Detection Service Started")

for msg in consumer:

    try:

        transaction = msg.value

        features = [[
            transaction["amount"],
            transaction["merchant_score"],
            transaction["hour"]
        ]]

        fraud_prediction = xgb_model.predict(features)[0]
        anomaly_prediction = iso_model.predict(features)[0]

        if fraud_prediction == 1 or anomaly_prediction == -1:

            send_alert(transaction)
        else:
            print("Legitimate Transaction:", transaction)
    except Exception as e:
        print("Error:", e)
        producer.send(
            DLQ_TOPIC,
            {
                "failed_message": transaction,
                "error": str(e)
            }
        )
