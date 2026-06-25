import os
import random
import datetime
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

from src.features import compute_features, FEATURE_COLUMNS
from src.db import init_db

# Ensure models directory exists
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))
os.makedirs(MODELS_DIR, exist_ok=True)

def generate_synthetic_history(num_users: int = 100, txs_per_user: int = 50) -> pd.DataFrame:
    """
    Generates realistic historical transactions for a set of users, sorted by time.
    """
    print(f"Generating synthetic transactions for {num_users} users...")
    
    # Coordinates of some US cities to simulate realistic geolocations
    locations = [
        {"city": "New York", "lat": 40.7128, "lon": -74.0060},
        {"city": "Los Angeles", "lat": 34.0522, "lon": -118.2437},
        {"city": "Chicago", "lat": 41.8781, "lon": -87.6298},
        {"city": "Houston", "lat": 29.7604, "lon": -95.3698},
        {"city": "Phoenix", "lat": 33.4484, "lon": -112.0740},
        {"city": "Philadelphia", "lat": 39.9526, "lon": -75.1652},
        {"city": "San Antonio", "lat": 29.4241, "lon": -98.4936},
        {"city": "San Diego", "lat": 32.7157, "lon": -117.1611},
        {"city": "Dallas", "lat": 32.7767, "lon": -96.7970},
        {"city": "San Jose", "lat": 37.3382, "lon": -121.8863}
    ]
    
    categories = ["retail", "grocery", "food", "entertainment", "travel", "utilities", "cash_withdrawal"]
    
    start_time = datetime.datetime.utcnow() - datetime.timedelta(days=10)
    tx_list = []
    
    # Generate users profile
    user_profiles = {}
    for i in range(num_users):
        u_id = f"user_{1000 + i}"
        base_loc = random.choice(locations)
        avg_amount = random.uniform(15.0, 120.0)
        user_profiles[u_id] = {
            "lat": base_loc["lat"],
            "lon": base_loc["lon"],
            "avg_amount": avg_amount
        }

    # Generate sequential transactions per user
    for u_id, profile in user_profiles.items():
        curr_time = start_time
        
        # User state trackers
        last_lat = profile["lat"]
        last_lon = profile["lon"]
        
        for tx_idx in range(txs_per_user):
            # Time delta: usually 1 to 12 hours
            time_gap = random.uniform(3600, 3600 * 12)
            curr_time += datetime.timedelta(seconds=time_gap)
            
            # Amount: log-normal distribution around user avg
            amount = float(np.random.lognormal(mean=np.log(profile["avg_amount"]), sigma=0.4))
            amount = max(1.0, round(amount, 2))
            
            # Location: slightly offset from base, occasionally travel
            if random.random() < 0.03:  # 3% travel chance
                travel_loc = random.choice(locations)
                lat = travel_loc["lat"] + random.uniform(-0.1, 0.1)
                lon = travel_loc["lon"] + random.uniform(-0.1, 0.1)
            else:
                lat = last_lat + random.uniform(-0.02, 0.02)
                lon = last_lon + random.uniform(-0.02, 0.02)
                
            last_lat, last_lon = lat, lon
            
            tx_id = f"tx_{u_id}_{tx_idx}"
            
            tx_list.append({
                "transaction_id": tx_id,
                "user_id": u_id,
                "amount": amount,
                "timestamp": curr_time.isoformat(),
                "lat": lat,
                "lon": lon,
                "merchant_category": random.choice(categories),
                "device_id": f"device_{u_id[:8]}",
                "ip_address": f"192.168.1.{random.randint(2, 254)}",
                "is_fraud_labeled": 0
            })
            
    # Convert to DataFrame
    df = pd.DataFrame(tx_list)
    # Sort chronologically globally
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp_dt").drop(columns=["timestamp_dt"]).reset_index(drop=True)
    
    # Inject fraud: ~2% fraud rate
    print("Injecting synthetic fraud scenarios...")
    num_fraud = int(len(df) * 0.02)
    fraud_indices = random.sample(range(100, len(df)), num_fraud)
    
    for idx in fraud_indices:
        df.loc[idx, "is_fraud_labeled"] = 1
        fraud_type = random.choice(["amount_spike", "velocity_spike", "geolocation_jump"])
        u_id = df.loc[idx, "user_id"]
        profile = user_profiles[u_id]
        
        if fraud_type == "amount_spike":
            # 10x-50x user avg amount
            df.loc[idx, "amount"] = round(profile["avg_amount"] * random.uniform(15.0, 40.0), 2)
            
        elif fraud_type == "velocity_spike":
            # Rapid high value transactions in a short time span
            # We modify previous transactions in sequence to create a cluster
            base_time = pd.to_datetime(df.loc[idx, "timestamp"])
            for offset_idx in range(1, 4):
                if idx - offset_idx >= 0:
                    prev_idx = idx - offset_idx
                    df.loc[prev_idx, "user_id"] = u_id
                    df.loc[prev_idx, "timestamp"] = (base_time - datetime.timedelta(seconds=random.uniform(30, 90) * offset_idx)).isoformat()
                    df.loc[prev_idx, "amount"] = round(profile["avg_amount"] * random.uniform(3.0, 7.0), 2)
                    df.loc[prev_idx, "is_fraud_labeled"] = 1
                    
        elif fraud_type == "geolocation_jump":
            # Transaction in a location very far from the user's recent location in a short time
            # Find a city far away
            current_city = locations[random.randint(0, len(locations)-1)]
            df.loc[idx, "lat"] = current_city["lat"]
            df.loc[idx, "lon"] = current_city["lon"]
            # Set time extremely close to the transaction immediately preceding it for that user
            user_txs = df[df["user_id"] == u_id]
            user_txs_before = user_txs[user_txs.index < idx]
            if not user_txs_before.empty:
                last_tx_before = user_txs_before.iloc[-1]
                last_time = pd.to_datetime(last_tx_before["timestamp"])
                df.loc[idx, "timestamp"] = (last_time + datetime.timedelta(seconds=random.uniform(30, 180))).isoformat()
                
    return df

def train_and_save_models():
    # 1. Generate historical transaction data
    df = generate_synthetic_history(num_users=80, txs_per_user=40)
    
    # Initialize DB with mock data to build up state
    init_db()
    
    # 2. Compute streaming features sequentially
    print("Computing real-time features on synthetic historical data...")
    features_list = []
    
    # We will simulate feature engineering by keeping track of history per user
    user_histories = {}
    
    for idx, row in df.iterrows():
        u_id = row["user_id"]
        tx_dict = row.to_dict()
        
        # Get history
        hist = user_histories.get(u_id, [])
        
        # Compute features
        feats = compute_features(tx_dict, hist)
        feats["is_fraud_labeled"] = row["is_fraud_labeled"]
        features_list.append(feats)
        
        # Update history
        hist.insert(0, tx_dict)  # Prepend latest
        user_histories[u_id] = hist
        
    feats_df = pd.DataFrame(features_list)
    
    # 3. Train Test Split
    print("Training models...")
    X = feats_df[FEATURE_COLUMNS]
    y = feats_df["is_fraud_labeled"]
    
    # We split chronologically or standard random (since this is synthetic)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    # Model 1: Isolation Forest (Unsupervised)
    # Train on normal transactions only, or all transactions (contamination set to estimated rate)
    contamination = float(y_train.mean())
    if contamination == 0:
        contamination = 0.02
    
    iforest = IsolationForest(
        n_estimators=150,
        max_samples='auto',
        contamination=contamination,
        random_state=42
    )
    iforest.fit(X_train)
    
    # Model 2: XGBoost Classifier (Supervised)
    # Adjust scale_pos_weight for class imbalance
    scale_pos_weight = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-5)
    
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss"
    )
    xgb.fit(X_train, y_train)
    
    # 4. Evaluation
    print("\n--- Isolation Forest Evaluation (Test Set) ---")
    # iForest returns -1 for anomaly, 1 for normal. Map to 1 and 0.
    iforest_preds = iforest.predict(X_test)
    iforest_preds_mapped = np.where(iforest_preds == -1, 1, 0)
    print(classification_report(y_test, iforest_preds_mapped))
    
    print("\n--- XGBoost Classifier Evaluation (Test Set) ---")
    xgb_probs = xgb.predict_proba(X_test)[:, 1]
    xgb_preds = (xgb_probs > 0.5).astype(int)
    print(classification_report(y_test, xgb_preds))
    print(f"XGBoost ROC AUC Score: {roc_auc_score(y_test, xgb_probs):.4f}")
    
    # 5. Save models
    iforest_path = os.path.join(MODELS_DIR, "isolation_forest.joblib")
    xgb_path = os.path.join(MODELS_DIR, "xgboost_model.joblib")
    
    joblib.dump(iforest, iforest_path)
    joblib.dump(xgb, xgb_path)
    
    print(f"\nModels successfully saved to:\n - {iforest_path}\n - {xgb_path}")

if __name__ == "__main__":
    train_and_save_models()
