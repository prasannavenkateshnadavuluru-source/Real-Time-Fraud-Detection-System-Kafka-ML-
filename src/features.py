import math
import datetime
from typing import Dict, List, Any
import numpy as np

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes the Haversine distance in kilometers between two points.
    """
    R = 6371.0  # Earth's radius in kilometers
    
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c

def parse_iso_timestamp(ts_str: str) -> datetime.datetime:
    # Try parsing standard ISO formats (e.g. 2026-06-11T15:19:01 or 2026-06-11 15:19:01)
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            return datetime.datetime.strptime(ts_str.split("+")[0], fmt)
        except ValueError:
            continue
    return datetime.datetime.utcnow()

def compute_features(tx: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes real-time features for a transaction based on the user's history.
    history is expected to be a list of past transactions for this user, sorted from newest to oldest.
    """
    current_time = parse_iso_timestamp(tx['timestamp'])
    current_lat = float(tx['lat'])
    current_lon = float(tx['lon'])
    current_amount = float(tx['amount'])
    
    # 1. Base Feature: hour_of_day
    hour_of_day = current_time.hour
    
    # Initialize variables for aggregates
    velocity_10m = 0
    velocity_1h = 0
    amounts_1h = []
    time_delta_seconds = 86400.0  # Default to 24 hours if no prior tx
    distance_delta_km = 0.0
    
    if history:
        # The history is sorted newest to oldest, so history[0] is the most recent past transaction
        most_recent = history[0]
        mr_time = parse_iso_timestamp(most_recent['timestamp'])
        
        # Time delta
        td = (current_time - mr_time).total_seconds()
        time_delta_seconds = max(0.0, td)  # Avoid negative deltas due to clock drift simulation
        
        # Distance delta
        mr_lat = float(most_recent['lat'])
        mr_lon = float(most_recent['lon'])
        distance_delta_km = haversine_distance(current_lat, current_lon, mr_lat, mr_lon)
        
        # Rolling aggregates
        for past_tx in history:
            past_time = parse_iso_timestamp(past_tx['timestamp'])
            diff_seconds = (current_time - past_time).total_seconds()
            
            if diff_seconds < 0:
                continue  # Skip future records if database contains any
                
            if diff_seconds <= 600:  # 10 minutes
                velocity_10m += 1
                
            if diff_seconds <= 3600:  # 1 hour
                velocity_1h += 1
                amounts_1h.append(float(past_tx['amount']))
    
    # Amount ratio in the last 1 hour
    if amounts_1h:
        mean_amount_1h = np.mean(amounts_1h)
        amount_ratio_1h = current_amount / (mean_amount_1h + 1e-5)
    else:
        amount_ratio_1h = 1.0  # Default when no history in the last hour
        
    return {
        "amount": current_amount,
        "hour_of_day": hour_of_day,
        "velocity_10m": velocity_10m,
        "velocity_1h": velocity_1h,
        "amount_ratio_1h": float(amount_ratio_1h),
        "time_delta_seconds": float(time_delta_seconds),
        "distance_delta_km": float(distance_delta_km)
    }

# Feature names in the exact order the models expect
FEATURE_COLUMNS = [
    "amount",
    "hour_of_day",
    "velocity_10m",
    "velocity_1h",
    "amount_ratio_1h",
    "time_delta_seconds",
    "distance_delta_km"
]

def to_feature_vector(features_dict: Dict[str, Any]) -> List[float]:
    return [features_dict[col] for col in FEATURE_COLUMNS]
