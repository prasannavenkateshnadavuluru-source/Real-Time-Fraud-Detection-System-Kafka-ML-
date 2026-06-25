import os
import joblib
from typing import Dict, Any, Tuple
import numpy as np

from src.features import to_feature_vector, FEATURE_COLUMNS

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
IFOREST_PATH = os.path.join(MODELS_DIR, "isolation_forest.joblib")
XGB_PATH = os.path.join(MODELS_DIR, "xgboost_model.joblib")

class ModelManager:
    def __init__(self):
        self.xgb = None
        self.iforest = None
        self.load_models()

    def load_models(self) -> bool:
        if os.path.exists(IFOREST_PATH) and os.path.exists(XGB_PATH):
            try:
                self.iforest = joblib.load(IFOREST_PATH)
                self.xgb = joblib.load(XGB_PATH)
                print("Models loaded successfully.")
                return True
            except Exception as e:
                print(f"Error loading models: {e}")
                return False
        else:
            print("Models not found on disk.")
            return False

    def is_ready(self) -> bool:
        return self.xgb is not None and self.iforest is not None

    def predict(self, features_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs inference using both XGBoost and Isolation Forest, combining their predictions.
        """
        if not self.is_ready():
            # Fallback mock scoring if models are not trained yet
            return self._mock_predict(features_dict)
            
        # Format features into list/vector
        vec = [to_feature_vector(features_dict)]
        
        # XGBoost prediction (probability of class 1)
        xgb_prob = float(self.xgb.predict_proba(vec)[0, 1])
        
        # Isolation Forest score
        # decision_function: average anomaly score of X. Opposing of the anomaly score (lower is more anomalous).
        # Normal observations yield positive decision scores, anomalies yield negative.
        iforest_decision = float(self.iforest.decision_function(vec)[0])
        
        # Map iForest decision score to a 0-1 anomaly score for display
        # Typical range of decision_function is around -0.5 to +0.5
        # Lower score = more anomalous. Let's make an "anomaly score" from 0 to 1
        # where 1 is highly anomalous.
        iforest_anomaly = float(1.0 / (1.0 + np.exp(iforest_decision * 10))) # Sigmoid mapping
        
        # Combined logic
        # High confidence fraud: supervised probability > 0.7 OR high anomaly + medium supervised
        is_fraud = int(xgb_prob > 0.6 or (xgb_prob > 0.4 and iforest_decision < -0.05))
        
        # Determine alert level
        if xgb_prob >= 0.8 or (xgb_prob >= 0.6 and iforest_decision < -0.05):
            alert_level = "HIGH"
        elif xgb_prob >= 0.4 or iforest_decision < 0.0:
            alert_level = "MEDIUM"
        else:
            alert_level = "LOW"
            
        return {
            "xgb_prob": xgb_prob,
            "iforest_score": iforest_anomaly,
            "iforest_decision": iforest_decision,
            "is_fraud": is_fraud,
            "alert_level": alert_level
        }

    def _mock_predict(self, features_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        A heuristic-based fallback classification when ML models are not yet trained.
        """
        amount = features_dict.get("amount", 0.0)
        velocity_10m = features_dict.get("velocity_10m", 0)
        distance = features_dict.get("distance_delta_km", 0.0)
        time_delta = features_dict.get("time_delta_seconds", 86400.0)
        
        # Simple rule-based mock
        score = 0.0
        if amount > 1000:
            score += 0.4
        if velocity_10m >= 3:
            score += 0.3
        if distance > 500 and time_delta < 300: # Fast travel anomaly
            score += 0.5
            
        xgb_prob = min(0.99, score)
        iforest_score = min(0.99, (amount / 2000.0) if amount > 500 else 0.1)
        
        is_fraud = int(xgb_prob > 0.6)
        if xgb_prob >= 0.7:
            alert_level = "HIGH"
        elif xgb_prob >= 0.3:
            alert_level = "MEDIUM"
        else:
            alert_level = "LOW"
            
        return {
            "xgb_prob": xgb_prob,
            "iforest_score": iforest_score,
            "iforest_decision": -0.1 if is_fraud else 0.1,
            "is_fraud": is_fraud,
            "alert_level": alert_level
        }
