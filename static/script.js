// static/script.js

// ---------- DOM refs ----------
const refreshBtn = document.getElementById('refreshBtn');
const autoTradeToggle = document.getElementById('autoTradeToggle');
const autoTradeStatus = document.getElementById('autoTradeStatus');
const retrainBtn = document.getElementById('retrainBtn');
const retrainStatus = document.getElementById('retrainStatus');
const loading = document.getElementById('loading');
const errorMsg = document.getElementById('errorMsg');
const summary = document.getElementById('summary');

// ---------- Utilities ----------
function showLoading(show) {
    loading.classList.toggle('hidden', !show);
}

function showError(msg) {
    if (msg) {
        errorMsg.textContent = msg;
        errorMsg.classList.remove('hidden');
    } else {
        errorMsg.classList.add('hidden');
    }
}

function updateServerTime() {
    const now = new Date();
    document.getElementById('serverTime').textContent = now.toLocaleTimeString();
}
setInterval(updateServerTime, 1000);
updateServerTime();

// ---------- Fetch signals ----------
async function fetchSignals() {
    showLoading(true);
    showError(null);
    try {
        const resp = await fetch('/api/signals', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                pairs: 'EURUSD=X,GBPUSD=X,AUDUSD=X,USDCAD=X,USDCHF=X,EURGBP=X,EURJPY=X,NZDUSD=X,GBPJPY=X,USDJPY=X',
                interval: '1h',
                atr_period: 14,
                risk_mult: 1.0
            })
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        renderSignals(data);
    } catch (err) {
        showError('Failed to fetch signals: ' + err.message);
    } finally {
        showLoading(false);
    }
}

// ---------- Render signals table ----------
function renderSignals(signals) {
    summary.innerHTML = '';
    if (!signals || signals.length === 0) {
        summary.innerHTML = '<p style="color: var(--text-secondary);">No signals returned.</p>';
        return;
    }

    const table = document.createElement('table');
    table.className = 'signal-table';
    table.innerHTML = `
        <thead>
            <tr>
                <th>Pair</th>
                <th>Signal</th>
                <th>Confidence</th>
                <th>Price</th>
                <th>TP</th>
                <th>SL</th>
                <th>ATR</th>
                <th>Ready</th>
                <th>Action</th>
            </tr>
        </thead>
        <tbody></tbody>
    `;
    const tbody = table.querySelector('tbody');

    signals.forEach(sig => {
        const row = document.createElement('tr');

        // Pair
        const pairCell = document.createElement('td');
        pairCell.textContent = sig.pair || 'N/A';
        row.appendChild(pairCell);

        // Signal badge
        const signalCell = document.createElement('td');
        const badge = document.createElement('span');
        const sigText = sig.signal || 'HOLD';
        badge.className = `signal-${sigText}`;
        badge.textContent = sigText;
        signalCell.appendChild(badge);
        row.appendChild(signalCell);

        // Confidence
        const confCell = document.createElement('td');
        confCell.textContent = sig.confidence ? (sig.confidence * 100).toFixed(1) + '%' : '0%';
        row.appendChild(confCell);

        // Price
        const priceCell = document.createElement('td');
        priceCell.textContent = sig.price ? sig.price.toFixed(5) : '—';
        row.appendChild(priceCell);

        // TP
        const tpCell = document.createElement('td');
        tpCell.textContent = sig.tp ? sig.tp.toFixed(5) : '—';
        row.appendChild(tpCell);

        // SL
        const slCell = document.createElement('td');
        slCell.textContent = sig.sl ? sig.sl.toFixed(5) : '—';
        row.appendChild(slCell);

        // ATR
        const atrCell = document.createElement('td');
        atrCell.textContent = sig.atr ? sig.atr.toFixed(5) : '—';
        row.appendChild(atrCell);

        // Trade Ready (new column)
        const readyCell = document.createElement('td');
        const isReady = sig.trade_ready === true;
        const readyIcon = document.createElement('i');
        readyIcon.className = `fas fa-${isReady ? 'check-circle' : 'times-circle'} ready-icon ${isReady ? 'true' : 'false'}`;
        readyIcon.title = sig.trade_ready_reason || (isReady ? 'Ready' : 'Not ready');
        readyCell.appendChild(readyIcon);
        row.appendChild(readyCell);

        // Action button
        const actionCell = document.createElement('td');
        const tradeBtn = document.createElement('button');
        tradeBtn.textContent = 'Trade Now';
        tradeBtn.className = 'trade-btn';

        const canTrade = sig.signal !== 'HOLD' && sig.can_trade === true;
        const ready = sig.trade_ready === true;

        if (canTrade && ready) {
            tradeBtn.style.background = '#10b981'; // green
            tradeBtn.onclick = () => executeTrade(sig);
        } else {
            tradeBtn.style.background = '#6b7280'; // grey
            tradeBtn.disabled = true;
            let reason = sig.trade_ready_reason || 'Conditions not met';
            if (!canTrade) reason = sig.can_trade_reason || 'Invalid SL/TP or no signal';
            tradeBtn.title = reason;
        }
        actionCell.appendChild(tradeBtn);
        row.appendChild(actionCell);

        tbody.appendChild(row);
    });

    summary.appendChild(table);
}

// ---------- Execute trade (manual) ----------
async function executeTrade(signalData) {
    if (!confirm(`Execute ${signalData.signal} on ${signalData.pair} at ${signalData.price}?`)) return;
    try {
        const resp = await fetch('/api/autotrade', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                pairs: signalData.pair + '=X',
                interval: '1h',
                atr_period: 14,
                risk_mult: 1.0,
                volume: 0.01
            })
        });
        const result = await resp.json();
        // ✅ FIXED: no optional chaining
        if (resp.ok && result[0] && result[0].trade && result[0].trade.success) {
            alert(`✅ Trade executed! Order ID: ${result[0].trade.order_id}`);
        } else {
            let errorMsg = 'Unknown error';
            if (result[0] && result[0].trade && result[0].trade.error) {
                errorMsg = result[0].trade.error;
            }
            alert(`❌ Trade failed: ${errorMsg}`);
        }
    } catch (err) {
        alert('Error executing trade: ' + err.message);
    }
}

// ---------- Auto‑trade toggle ----------
async function setAutoTrade(enabled) {
    try {
        const resp = await fetch('/api/auto_trade_status', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ enabled })
        });
        const data = await resp.json();
        autoTradeStatus.textContent = data.enabled ? 'Enabled' : 'Disabled';
    } catch (err) {
        console.error('Auto‑trade toggle error:', err);
    }
}

autoTradeToggle.addEventListener('change', function() {
    setAutoTrade(this.checked);
});

// ---------- Retrain model ----------
retrainBtn.addEventListener('click', async function() {
    retrainStatus.textContent = '⏳ Retraining...';
    retrainBtn.disabled = true;
    try {
        const resp = await fetch('/api/retrain', { method: 'POST' });
        const data = await resp.json();
        retrainStatus.textContent = data.success ? '✅ Done' : '❌ Failed';
    } catch (err) {
        retrainStatus.textContent = '❌ Error';
    } finally {
        retrainBtn.disabled = false;
        setTimeout(() => { retrainStatus.textContent = ''; }, 5000);
    }
});

// ---------- Refresh ----------
refreshBtn.addEventListener('click', fetchSignals);

// ---------- Load initial signals ----------
fetchSignals();

// Also fetch auto‑trade status on load
(async function getAutoTradeStatus() {
    try {
        const resp = await fetch('/api/auto_trade_status');
        const data = await resp.json();
        autoTradeToggle.checked = data.enabled;
        autoTradeStatus.textContent = data.enabled ? 'Enabled' : 'Disabled';
    } catch (_) { /* ignore */ }
})();