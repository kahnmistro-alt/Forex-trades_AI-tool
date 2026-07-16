document.addEventListener('DOMContentLoaded', () => {
            const refreshBtn = document.getElementById('refreshBtn');
            const loadingDiv = document.getElementById('loading');
            const errorDiv = document.getElementById('errorMsg');
            const summaryDiv = document.getElementById('summary');

            const autoTradeToggle = document.getElementById('autoTradeToggle');
            const autoTradeStatus = document.getElementById('autoTradeStatus');

            if (autoTradeToggle) {
                fetch('/api/auto_trade_status')
                    .then(res => res.json())
                    .then(data => {
                        autoTradeToggle.checked = data.enabled;
                        autoTradeStatus.textContent = data.enabled ? 'Enabled (60s)' : 'Disabled';
                        autoTradeStatus.style.color = data.enabled ? '#10b981' : 'var(--text-secondary)';
                    });

                autoTradeToggle.addEventListener('change', () => {
                    const enabled = autoTradeToggle.checked;
                    const fixedPairs = 'EURUSD=X, GBPUSD=X, USDJPY=X, AUDUSD=X, USDCAD=X';
                    if (enabled) {
                        fetch('/api/auto_trade_pairs', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ pairs: fixedPairs })
                        });
                    }
                    fetch('/api/auto_trade_status', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ enabled: enabled })
                        })
                        .then(res => res.json())
                        .then(data => {
                            autoTradeStatus.textContent = data.enabled ? 'Enabled (60s)' : 'Disabled';
                            autoTradeStatus.style.color = data.enabled ? '#10b981' : 'var(--text-secondary)';
                        });
                });
            }

            function updateServerTime() {
                document.getElementById('serverTime').innerHTML = `<i class="far fa-clock"></i> ${new Date().toLocaleString()}`;
            }
            updateServerTime();
            setInterval(updateServerTime, 1000);

            refreshBtn.addEventListener('click', refreshSignals);

            async function refreshSignals() {
                loadingDiv.classList.remove('hidden');
                errorDiv.classList.add('hidden');
                summaryDiv.innerHTML = '';

                const pairs = 'EURUSD=X, GBPUSD=X, USDJPY=X, AUDUSD=X, USDCAD=X';
                const payload = {
                    pairs: pairs,
                    interval: '1h',
                    atr_period: 14,
                    risk_mult: 1.0
                };

                try {
                    const response = await fetch('/api/signals', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });
                    const results = await response.json();
                    loadingDiv.classList.add('hidden');

                    if (!results.length) {
                        errorDiv.textContent = 'No data returned.';
                        errorDiv.classList.remove('hidden');
                        return;
                    }

                    const valid = results.filter(r => !r.error);
                    const errors = results.filter(r => r.error);
                    if (errors.length) {
                        errorDiv.innerHTML = `<i class="fas fa-exclamation-triangle"></i> ${errors.map(e => `${e.pair}: ${e.error}`).join('; ')}`;
                errorDiv.classList.remove('hidden');
            }

            renderSummary(valid);
        } catch (err) {
            loadingDiv.classList.add('hidden');
            errorDiv.textContent = 'Network error: ' + err.message;
            errorDiv.classList.remove('hidden');
        }
    }

    function renderSummary(results) {
        if (!results.length) return;
        const title = document.createElement('h2');
        title.innerHTML = '<i class="fas fa-table-list"></i> Live Signals';
        summaryDiv.appendChild(title);

        const table = document.createElement('table');
        table.className = 'signal-table';
        table.innerHTML = `
            <thead>
                <tr><th>Pair</th><th>Signal</th><th>Conf</th><th>Pattern(s)</th><th>Trend</th><th>Price</th><th>TP</th><th>SL</th><th>ATR</th><th>Action</th><th>Trade</th></tr>
            </thead>
            <tbody></tbody>
        `;
        const tbody = table.querySelector('tbody');
        results.forEach(r => {
            const row = tbody.insertRow();
            row.dataset.pair = r.pair;
            row.dataset.signal = r.signal;
            row.dataset.price = r.price;
            row.dataset.tp = r.tp;
            row.dataset.sl = r.sl;
            row.dataset.canTrade = r.can_trade;

            row.insertCell(0).textContent = r.pair;
            const sigCell = row.insertCell(1);
            sigCell.innerHTML = `<span class="signal-${r.signal}">${r.signal}</span>`;
            row.insertCell(2).textContent = r.confidence;

            const patterns = r.pattern_details ? Object.keys(r.pattern_details.candle_patterns || {}).join(', ') : 'None';
            row.insertCell(3).textContent = patterns;

            row.insertCell(4).textContent = r.trend || 'neutral';
            row.insertCell(5).textContent = r.price;
            row.insertCell(6).textContent = r.tp;
            row.insertCell(7).textContent = r.sl;
            row.insertCell(8).textContent = r.atr;

            const copyCell = row.insertCell(9);
            const copyBtn = document.createElement('button');
            copyBtn.textContent = '📋 Copy';
            copyBtn.className = 'copy-btn';
            copyBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                copyTradePlan(r);
            });
            copyCell.appendChild(copyBtn);

            const tradeCell = row.insertCell(10);
            if (r.signal !== 'HOLD') {
                const tradeBtn = document.createElement('button');
                tradeBtn.textContent = '🚀 Trade Now';
                if (r.can_trade) {
                    tradeBtn.className = 'trade-btn';
                } else {
                    tradeBtn.className = 'trade-btn trade-btn-disabled';
                    tradeBtn.disabled = true;
                    tradeBtn.title = r.can_trade_reason || 'Not valid';
                }
                tradeBtn.dataset.pair = r.pair;
                tradeBtn.dataset.signal = r.signal;
                tradeBtn.dataset.price = r.price;
                tradeBtn.dataset.tp = r.tp;
                tradeBtn.dataset.sl = r.sl;
                tradeBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    if (!tradeBtn.disabled) {
                        tradeNow(r.pair, r.signal, r.price, r.tp, r.sl);
                    }
                });
                tradeCell.appendChild(tradeBtn);
            } else {
                tradeCell.textContent = '—';
            }
        });
        summaryDiv.appendChild(table);
    }

    async function tradeNow(pair, signal, price, tp, sl) {
        const payload = {
            pairs: pair + '=X',
            interval: '1h',
            atr_period: 14,
            risk_mult: 1.0,
            volume: 0.01
        };
        try {
            const resp = await fetch('/api/autotrade', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const result = await resp.json();
            if (result[0]?.trade?.success) {
                alert(`✅ ${result[0].trade.message}`);
            } else {
                alert(`❌ ${result[0]?.trade?.error || 'Failed'}`);
            }
        } catch (err) {
            alert('❌ ' + err.message);
        }
    }

    function copyTradePlan(r) {
        const text = `${r.signal} ${r.pair} at ${r.price}\nTP: ${r.tp}\nSL: ${r.sl}`;
        navigator.clipboard.writeText(text);
    }

    window.tradeNow = tradeNow;
});