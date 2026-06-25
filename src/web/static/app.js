// Application Dashboard Logic - Real-Time Fraud Detection System

let socket = null;
let throughputChart = null;
const maxChartPoints = 15;
let chartLabels = [];
let chartProcessedData = [];
let chartFraudData = [];

// Initialize Dashboard
document.addEventListener("DOMContentLoaded", () => {
    initChart();
    connectWebSocket();
    setupEventListeners();
});

// Setup Chart.js
function initChart() {
    const ctx = document.getElementById('throughput-chart').getContext('2d');
    
    throughputChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartLabels,
            datasets: [
                {
                    label: 'Normal Processed',
                    data: chartProcessedData,
                    borderColor: '#0ea5e9',
                    backgroundColor: 'rgba(14, 165, 233, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 2
                },
                {
                    label: 'Flagged Fraud',
                    data: chartFraudData,
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.15)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        color: '#94a3b8',
                        font: { family: 'Inter', size: 10 }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.03)' },
                    ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 9 } }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.03)' },
                    ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 9 } },
                    suggestedMin: 0,
                    suggestedMax: 10
                }
            }
        }
    });
}

// Update Chart Data over time
function updateChart(metrics) {
    const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    
    // Total processed in this second (approx change)
    const totalProcessed = metrics.total_processed;
    const totalFraud = metrics.total_fraud;
    
    chartLabels.push(now);
    chartProcessedData.push(totalProcessed);
    chartFraudData.push(totalFraud);
    
    if (chartLabels.length > maxChartPoints) {
        chartLabels.shift();
        chartProcessedData.shift();
        chartFraudData.shift();
    }
    
    throughputChart.update();
}

// WebSocket Connection
function connectWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/ws`;
    
    const statusText = document.getElementById("connection-status");
    const statusDot = document.querySelector(".status-indicator-dot");
    
    socket = new WebSocket(wsUrl);
    
    socket.onopen = () => {
        statusText.innerText = "STREAMING PIPELINE ACTIVE";
        statusDot.className = "status-indicator-dot online";
        logger("WebSocket connection established.");
    };
    
    socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        updateKPIs(data.metrics);
        updateTransactionsList(data.transactions);
        updateAlertsList(data.alerts);
        updateDLQList(data.dlq);
        updateSettingsDisplay(data.settings);
        updateChart(data.metrics);
    };
    
    socket.onclose = () => {
        statusText.innerText = "PIPELINE OFFLINE - RETRYING...";
        statusDot.className = "status-indicator-dot offline";
        logger("WebSocket connection closed. Retrying in 3 seconds...");
        setTimeout(connectWebSocket, 3000);
    };
    
    socket.onerror = (err) => {
        logger("WebSocket error: " + err);
    };
}

// Update KPI Panels
function updateKPIs(metrics) {
    document.getElementById("val-processed").innerText = metrics.total_processed.toLocaleString();
    document.getElementById("val-fraud-rate").innerText = `${metrics.fraud_rate_pct.toFixed(2)}%`;
    document.getElementById("val-fraud-count").innerText = `${metrics.total_fraud} Flagged`;
    document.getElementById("val-retries").innerText = metrics.active_retries;
    document.getElementById("val-dlq").innerText = metrics.dlq_count;
    document.getElementById("val-xgb").innerText = metrics.avg_xgb_score.toFixed(3);
    document.getElementById("val-iforest").innerText = metrics.avg_iforest_score.toFixed(3);
    
    // Highlight KPI red if active DLQ backlog exists
    const dlqCard = document.getElementById("kpi-dlq");
    if (metrics.dlq_count > 0) {
        dlqCard.classList.add("dlq-active");
    } else {
        dlqCard.classList.remove("dlq-active");
    }
}

// Update Transaction List Panel
function updateTransactionsList(txs) {
    const listElement = document.getElementById("transaction-list");
    if (!txs || txs.length === 0) {
        listElement.innerHTML = `<div class="no-data">Awaiting transaction stream...</div>`;
        return;
    }
    
    let html = "";
    txs.forEach(tx => {
        const isFraud = tx.is_fraud_predicted === 1;
        const isRetrying = tx.status === "RETRYING";
        const isDlq = tx.status === "DLQ";
        
        let txClass = "tx-item";
        if (isFraud) txClass += " tx-fraudulent";
        else if (isRetrying) txClass += " tx-retrying";
        else if (isDlq) txClass += " tx-dlq-failed";
        
        let statusBadge = "";
        if (isRetrying) {
            statusBadge = `<span class="tx-status-badge badge-retrying">RETRY ATTEMPT ${tx.retry_count}</span>`;
        } else if (isDlq) {
            statusBadge = `<span class="tx-status-badge badge-dlq">ROUTED TO DLQ</span>`;
        } else if (isFraud) {
            statusBadge = `<span class="tx-status-badge badge-processed text-red">FRAUD DETECTED</span>`;
        } else {
            statusBadge = `<span class="tx-status-badge badge-processed">PROCESSED</span>`;
        }

        const dateStr = new Date(tx.timestamp).toLocaleTimeString();
        
        // Highlight high scores
        const xgbClass = tx.xgb_score > 0.6 ? "score-high" : "";
        const iforestClass = tx.iforest_score > 0.6 ? "score-high" : "";
        
        html += `
            <div class="${txClass}" id="tx-${tx.transaction_id}">
                <div class="tx-header">
                    <span class="tx-user">${tx.user_id}</span>
                    <span class="tx-amount">$${tx.amount.toFixed(2)}</span>
                </div>
                <div class="tx-meta">
                    <span>${tx.merchant_category.toUpperCase()} // Loc: (${tx.lat.toFixed(3)}, ${tx.lon.toFixed(3)})</span>
                    <span>${dateStr}</span>
                </div>
                <div class="tx-details">
                    <span class="tx-score-tag ${xgbClass}">XGB: ${tx.xgb_score.toFixed(3)}</span>
                    <span class="tx-score-tag ${iforestClass}">iForest Anomaly: ${tx.iforest_score.toFixed(3)}</span>
                </div>
                ${statusBadge}
            </div>
        `;
    });
    
    listElement.innerHTML = html;
}

// Update Alerts List Panel
function updateAlertsList(alerts) {
    const listElement = document.getElementById("alerts-list");
    if (!alerts || alerts.length === 0) {
        listElement.innerHTML = `<div class="no-data">No active fraud alerts. System secure.</div>`;
        return;
    }
    
    let html = "";
    alerts.forEach(alert => {
        const dateStr = new Date(alert.timestamp).toLocaleTimeString();
        const alertLvlClass = alert.alert_level === "HIGH" ? "alert-lvl-high" : "alert-lvl-med";
        
        html += `
            <div class="alert-item">
                <div class="alert-header-line">
                    <span class="${alertLvlClass}">[${alert.alert_level} INCIDENT]</span>
                    <span>TX: ${alert.transaction_id}</span>
                </div>
                <div class="alert-desc">${alert.details}</div>
                <span class="alert-time">${dateStr}</span>
            </div>
        `;
    });
    
    listElement.innerHTML = html;
}

// Update Dead-Letter Queue Panel
function updateDLQList(dlqs) {
    const listElement = document.getElementById("dlq-list");
    const counterTag = document.getElementById("dlq-counter-tag");
    
    counterTag.innerText = `${dlqs.length} PENDING`;
    
    if (!dlqs || dlqs.length === 0) {
        listElement.innerHTML = `<div class="no-data">Dead-Letter Queue is empty. No failed transactions.</div>`;
        return;
    }
    
    let html = "";
    dlqs.forEach(item => {
        const dateStr = new Date(item.timestamp).toLocaleTimeString();
        
        html += `
            <div class="dlq-item" id="dlq-${item.transaction_id}">
                <div class="dlq-meta-row">
                    <span>TX: <strong>${item.transaction_id}</strong></span>
                    <span>${dateStr}</span>
                </div>
                <div class="dlq-error">
                    <strong>Error:</strong> ${item.error_reason}
                </div>
                <div class="dlq-payload-container">
                    <pre class="dlq-payload-raw">${item.raw_payload}</pre>
                </div>
                <button class="btn-reprocess" onclick="reprocessDLQ('${item.transaction_id}')">
                    REPROCESS EVENT
                </button>
            </div>
        `;
    });
    
    listElement.innerHTML = html;
}

// Reprocess Single DLQ transaction
function reprocessDLQ(txId) {
    logger(`Requesting reprocessing of DLQ event: ${txId}`);
    fetch(`/api/dlq/reprocess/${txId}`, { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                logger(`DLQ event ${txId} reprocessed successfully.`);
            } else {
                alert(`Error reprocessing transaction: ${data.message}`);
            }
        })
        .catch(err => console.error("Reprocess request failed", err));
}

// Update Control display states (e.g. database failures indicator, slider value)
let retrainingInterval = null;
function updateSettingsDisplay(settings) {
    // 1. Fault tolerance button state
    const faultBtn = document.getElementById("btn-toggle-fault");
    if (settings.simulate_db_failure) {
        faultBtn.innerText = "🔌 FAULT SIMULATION: ACTIVE";
        faultBtn.classList.add("btn-active");
    } else {
        faultBtn.innerText = "🔌 FAULT SIMULATION: OFF";
        faultBtn.classList.remove("btn-active");
    }
    
    // 2. Speed Slider text
    document.getElementById("speed-display").innerText = `${settings.generator_speed.toFixed(1)}s`;
    document.getElementById("speed-slider").value = settings.generator_speed;
    
    // 3. Retraining Toast handler
    const toast = document.getElementById("retrain-toast");
    const retrainBtn = document.getElementById("btn-retrain");
    
    if (!settings.models_ready) {
        // Models missing or retraining
        toast.classList.remove("hidden");
        document.getElementById("toast-text").innerText = "MODELS PRE-TRAINING INITIALIZING...";
        retrainBtn.disabled = true;
    } else {
        // Models ready
        if (retrainingInterval) {
            // We were retraining, clear interval and hide toast
            clearTimeout(retrainingInterval);
            retrainingInterval = null;
            toast.classList.add("hidden");
            retrainBtn.disabled = false;
        }
    }
}

// Setup Event Listeners
function setupEventListeners() {
    // 1. Inject Fraud Burst
    document.getElementById("btn-fraud-burst").addEventListener("click", () => {
        fetch("/api/control/fraud_burst", { method: 'POST' })
            .then(() => logger("Injected fraud burst. Check streams."));
    });
    
    // 2. Toggle DB Fault
    document.getElementById("btn-toggle-fault").addEventListener("click", () => {
        fetch("/api/control/toggle_fault", { method: 'POST' });
    });
    
    // 3. Retrain Pipeline
    const retrainBtn = document.getElementById("btn-retrain");
    const toast = document.getElementById("retrain-toast");
    
    retrainBtn.addEventListener("click", () => {
        retrainBtn.disabled = true;
        toast.classList.remove("hidden");
        document.getElementById("toast-text").innerText = "Retraining Models... Pipeline paused.";
        
        fetch("/api/control/retrain", { method: 'POST' })
            .then(() => {
                logger("Model retraining started.");
                // Set safety timeout in case ws fails
                retrainingInterval = setTimeout(() => {
                    toast.classList.add("hidden");
                    retrainBtn.disabled = false;
                }, 8000);
            });
    });
    
    // 4. Reset Data
    document.getElementById("btn-clear-db").addEventListener("click", () => {
        if (confirm("Are you sure you want to clear transaction statistics and reset Kafka offsets?")) {
            fetch("/api/control/clear_db", { method: 'POST' })
                .then(() => {
                    logger("Cleared cockpit DB & reset streams.");
                    chartLabels.length = 0;
                    chartProcessedData.length = 0;
                    chartFraudData.length = 0;
                    throughputChart.update();
                });
        }
    });
    
    // 5. Speed Slider
    const slider = document.getElementById("speed-slider");
    slider.addEventListener("input", (e) => {
        const val = parseFloat(e.target.value);
        document.getElementById("speed-display").innerText = `${val.toFixed(1)}s`;
    });
    
    slider.addEventListener("change", (e) => {
        const val = parseFloat(e.target.value);
        fetch("/api/control/speed", {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sleep_interval_sec: val })
        });
    });
}

function logger(msg) {
    console.log(`[Cockpit Log] ${msg}`);
}
