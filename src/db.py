import sqlite3
import json
import os
from typing import Dict, List, Any, Optional

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "fraud_detection.db"))

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Transactions table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        transaction_id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        amount REAL NOT NULL,
        timestamp TEXT NOT NULL,
        lat REAL NOT NULL,
        lon REAL NOT NULL,
        merchant_category TEXT,
        device_id TEXT,
        ip_address TEXT,
        is_fraud_labeled INTEGER DEFAULT 0,
        is_fraud_predicted INTEGER DEFAULT 0,
        xgb_score REAL DEFAULT 0.0,
        iforest_score REAL DEFAULT 0.0,
        status TEXT DEFAULT 'PROCESSED',
        retry_count INTEGER DEFAULT 0
    )
    """)
    
    # Index for fast real-time rolling aggregates
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tx_user_time ON transactions (user_id, timestamp)")
    
    # Alerts table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS alerts (
        alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
        transaction_id TEXT UNIQUE,
        timestamp TEXT NOT NULL,
        alert_level TEXT NOT NULL, -- LOW, MEDIUM, HIGH
        details TEXT,
        is_resolved INTEGER DEFAULT 0,
        FOREIGN KEY (transaction_id) REFERENCES transactions (transaction_id)
    )
    """)
    
    # DLQ Events table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS dlq_events (
        transaction_id TEXT PRIMARY KEY,
        timestamp TEXT NOT NULL,
        raw_payload TEXT NOT NULL,
        error_reason TEXT NOT NULL,
        status TEXT DEFAULT 'PENDING' -- PENDING, REPROCESSED
    )
    """)
    
    conn.commit()
    conn.close()

def save_transaction(tx: Dict[str, Any], is_fraud_predicted: int = 0, xgb_score: float = 0.0, iforest_score: float = 0.0, status: str = 'PROCESSED', retry_count: int = 0):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT OR REPLACE INTO transactions (
        transaction_id, user_id, amount, timestamp, lat, lon, merchant_category, device_id, ip_address, 
        is_fraud_labeled, is_fraud_predicted, xgb_score, iforest_score, status, retry_count
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        tx['transaction_id'], tx['user_id'], tx['amount'], tx['timestamp'], tx['lat'], tx['lon'],
        tx.get('merchant_category', ''), tx.get('device_id', ''), tx.get('ip_address', ''),
        tx.get('is_fraud_labeled', 0), is_fraud_predicted, xgb_score, iforest_score, status, retry_count
    ))
    conn.commit()
    conn.close()

def save_alert(tx_id: str, alert_level: str, details: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    import datetime
    now = datetime.datetime.utcnow().isoformat()
    try:
        cursor.execute("""
        INSERT INTO alerts (transaction_id, timestamp, alert_level, details, is_resolved)
        VALUES (?, ?, ?, ?, 0)
        """, (tx_id, now, alert_level, details))
        conn.commit()
    except sqlite3.IntegrityError:
        # Already logged
        pass
    finally:
        conn.close()

def save_dlq(tx_id: str, raw_payload: str, error_reason: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    import datetime
    now = datetime.datetime.utcnow().isoformat()
    cursor.execute("""
    INSERT OR REPLACE INTO dlq_events (transaction_id, timestamp, raw_payload, error_reason, status)
    VALUES (?, ?, ?, ?, 'PENDING')
    """, (tx_id, now, raw_payload, error_reason))
    conn.commit()
    conn.close()

def get_dlq_events() -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM dlq_events WHERE status = 'PENDING' ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    events = [dict(r) for r in rows]
    conn.close()
    return events

def resolve_dlq_event(tx_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE dlq_events SET status = 'REPROCESSED' WHERE transaction_id = ?", (tx_id,))
    conn.commit()
    conn.close()

def get_recent_transactions(limit: int = 50) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT t.*, a.alert_level 
    FROM transactions t 
    LEFT JOIN alerts a ON t.transaction_id = a.transaction_id
    ORDER BY t.timestamp DESC 
    LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    txs = [dict(r) for r in rows]
    conn.close()
    return txs

def get_recent_alerts(limit: int = 20) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT a.*, t.user_id, t.amount, t.merchant_category, t.xgb_score, t.iforest_score
    FROM alerts a
    JOIN transactions t ON a.transaction_id = t.transaction_id
    ORDER BY a.timestamp DESC
    LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    alerts = [dict(r) for r in rows]
    conn.close()
    return alerts

def get_user_history(user_id: str, before_timestamp: str, limit: int = 100) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT amount, timestamp, lat, lon 
    FROM transactions 
    WHERE user_id = ? AND timestamp < ? AND status = 'PROCESSED'
    ORDER BY timestamp DESC 
    LIMIT ?
    """, (user_id, before_timestamp, limit))
    rows = cursor.fetchall()
    history = [dict(r) for r in rows]
    conn.close()
    return history

def get_system_metrics() -> Dict[str, Any]:
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Total processed
    cursor.execute("SELECT count(*) FROM transactions WHERE status = 'PROCESSED'")
    total_processed = cursor.fetchone()[0]
    
    # Total fraud predicted
    cursor.execute("SELECT count(*) FROM transactions WHERE is_fraud_predicted = 1 AND status = 'PROCESSED'")
    total_fraud = cursor.fetchone()[0]
    
    # Active retries
    cursor.execute("SELECT count(*) FROM transactions WHERE status = 'RETRYING'")
    active_retries = cursor.fetchone()[0]
    
    # DLQ count
    cursor.execute("SELECT count(*) FROM dlq_events WHERE status = 'PENDING'")
    dlq_count = cursor.fetchone()[0]
    
    # XGB avg score
    cursor.execute("SELECT avg(xgb_score) FROM transactions WHERE status = 'PROCESSED'")
    avg_xgb = cursor.fetchone()[0] or 0.0
    
    # iForest avg score
    cursor.execute("SELECT avg(iforest_score) FROM transactions WHERE status = 'PROCESSED'")
    avg_iforest = cursor.fetchone()[0] or 0.0

    conn.close()
    
    fraud_rate = (total_fraud / total_processed * 100) if total_processed > 0 else 0.0
    
    return {
        "total_processed": total_processed,
        "total_fraud": total_fraud,
        "fraud_rate_pct": round(fraud_rate, 2),
        "active_retries": active_retries,
        "dlq_count": dlq_count,
        "avg_xgb_score": round(avg_xgb, 4),
        "avg_iforest_score": round(avg_iforest, 4)
    }

def clear_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM transactions")
    cursor.execute("DELETE FROM alerts")
    cursor.execute("DELETE FROM dlq_events")
    conn.commit()
    conn.close()
