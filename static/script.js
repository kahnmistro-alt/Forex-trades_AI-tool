document.addEventListener('DOMContentLoaded', () => {
            // ---------- DOM references ----------
            const refreshBtn = document.getElementById('refreshBtn');
            const pairsInput = document.getElementById('pairs');
            const intervalSelect = document.getElementById('interval');
            const atrSlider = document.getElementById('atr_period');
            const atrValue = document.getElementById('atr_value');
            const riskSlider = document.getElementById('risk_mult');
            const riskValue = document.getElementById('risk_value');
            const loadingDiv = document.getElementById('loading');
            const errorDiv = document.getElementById('errorMsg');
            const summaryDiv = document.getElementById('summary');
            const actionableDiv = document.getElementById('actionable');
            const chartsDiv = document.getElementById('charts');
            const volumeInput = document.getElementById('volumeInput');

            // Auto-Trade elements
            const autoTradeToggle = document.getElementById('autoTradeToggle');
            const autoTradeStatus = document.getElementById('autoTradeStatus');

            // ---------- Auto-Trade toggle ----------
            if (autoTradeToggle) {
                // Load initial status
                fetch('/api/auto_trade_status')
                    .then(res => res.json())
                    .then(data => {
                        autoTradeToggle.checked = data.enabled;
                        autoTradeStatus.textContent = data.enabled ? 'Enabled (60s)' : 'Disabled';
                        autoTradeStatus.style.color = data.enabled ? '#10b981' : 'var(--text-secondary)';
                    })
                    .catch(err => console.error('Failed to load auto-trade status:', err));

                // Handle toggle change
                autoTradeToggle.addEventListener('change', () => {
                    const enabled = autoTradeToggle.checked;
                    // If enabling, send the current pair list to the backend
                    if (enabled) {
                        const pairs = pairsInput.value;
                        fetch('/api/auto_trade_pairs', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ pairs: pairs })
                            })
                            .then(res => res.json())
                            .then(data => console.log('Auto-trade pairs updated:', data))
                            .catch(err => console.error('Failed to set auto-trade pairs:', err));
                    }

                    // Set the enabled status
                    fetch('/api/auto_trade_status', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ enabled: enabled })
                        })
                        .then(res => res.json())
                        .then(data => {
                            autoTradeStatus.textContent = data.enabled ? 'Enabled (60s)' : 'Disabled';
                            autoTradeStatus.style.color = data.enabled ? '#10b981' : 'var(--text-secondary)';
                        })
                        .catch(err => console.error('Failed to toggle auto-trade:', err));
                });
            }

            // ---------- Sliders ----------
            atrSlider.addEventListener('input', () => { atrValue.textContent = atrSlider.value; });
            riskSlider.addEventListener('input', () => { riskValue.textContent = riskSlider.value; });

            // ---------- Server time ----------
            function updateServerTime() {
                const now = new Date();
                document.getElementById('serverTime').innerHTML = `<i class="far fa-clock"></i> ${now.toLocaleString()}`;
            }
            updateServerTime();
            setInterval(updateServerTime, 1000);

            // ---------- Refresh signals ----------
            refreshBtn.addEventListener('click', refreshSignals);

            async function refreshSignals() {
                loadingDiv.classList.remove('hidden');
                errorDiv.classList.add('hidden');
                summaryDiv.innerHTML = '';
                actionableDiv.innerHTML = '';
                chartsDiv.innerHTML = '';

                const payload = {
                    pairs: pairsInput.value,
                    interval: intervalSelect.value,
                    atr_period: parseInt(atrSlider.value),
                    risk_mult: parseFloat(riskSlider.value)
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
            renderActionableCards(valid);
            renderCharts(valid);
        } catch (err) {
            loadingDiv.classList.add('hidden');
            errorDiv.textContent = 'Network error: ' + err.message;
            errorDiv.classList.remove('hidden');
        }
    }

    // ---------- Render Summary Table ----------
    function renderSummary(results) {
        if (!results.length) return;
        const title = document.createElement('h2');
        title.innerHTML = '<i class="fas fa-table-list"></i> Live Signal Watchlist';
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
            const patterns = r.pattern_details ? Object.keys(r.pattern_details.patterns).join(', ') : 'None';
            row.insertCell(3).textContent = patterns;
            row.insertCell(4).textContent = r.trend || 'neutral';
            row.insertCell(5).textContent = r.price;
            row.insertCell(6).textContent = r.tp;
            row.insertCell(7).textContent = r.sl;
            row.insertCell(8).textContent = r.atr;

            // Copy
            const copyCell = row.insertCell(9);
            const copyBtn = document.createElement('button');
            copyBtn.textContent = '📋 Copy';
            copyBtn.className = 'copy-btn';
            copyBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                copyTradePlan(r);
            });
            copyCell.appendChild(copyBtn);

            // Trade
            const tradeCell = row.insertCell(10);
            if (r.signal !== 'HOLD') {
                const tradeBtn = document.createElement('button');
                tradeBtn.textContent = '🚀 Trade Now';
                if (r.can_trade) {
                    tradeBtn.className = 'trade-btn';
                } else {
                    tradeBtn.className = 'trade-btn trade-btn-disabled';
                    tradeBtn.disabled = true;
                    tradeBtn.title = r.can_trade_reason || 'Trade not possible';
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

    // ---------- Render Actionable Cards ----------
    function renderActionableCards(results) {
        const actionable = results.filter(r => r.signal !== 'HOLD');
        if (!actionable.length) {
            actionableDiv.innerHTML = '<div class="loading-card" style="text-align:center"><i class="fas fa-hourglass-half"></i> No actionable signals</div>';
            return;
        }
        const title = document.createElement('h2');
        title.innerHTML = '<i class="fas fa-bullhorn"></i> Ready-to-Trade Ideas';
        actionableDiv.appendChild(title);

        actionable.forEach(r => {
            const card = document.createElement('div');
            card.className = `trade-card trade-card-${r.signal.toLowerCase()}`;
            const patterns = r.pattern_details ? Object.keys(r.pattern_details.patterns).join(', ') : 'None';

            const infoDiv = document.createElement('div');
            infoDiv.className = 'trade-info';
            infoDiv.innerHTML = `
                <div class="trade-pair">${r.pair}</div>
                <div class="trade-signal">${r.signal} · confidence ${r.confidence}</div>
                <div class="trade-patterns">Patterns: ${patterns}</div>
            `;
            card.appendChild(infoDiv);

            const levelsDiv = document.createElement('div');
            levelsDiv.className = 'trade-levels';
            levelsDiv.innerHTML = `📈 TP: ${r.tp} &nbsp;|&nbsp; 📉 SL: ${r.sl}`;
            card.appendChild(levelsDiv);

            const actionsDiv = document.createElement('div');
            actionsDiv.className = 'trade-actions';

            // Copy
            const copyBtn = document.createElement('button');
            copyBtn.className = 'copy-btn copy-plan-btn';
            copyBtn.innerHTML = '<i class="far fa-copy"></i> Copy Plan';
            copyBtn.addEventListener('click', () => {
                const text = `${r.signal} ${r.pair} at ${r.price}\nTP: ${r.tp} (3:1 R:R)\nSL: ${r.sl}\nATR used: ${r.atr}`;
                navigator.clipboard.writeText(text);
                copyBtn.innerHTML = '<i class="fas fa-check"></i> Copied!';
                setTimeout(() => { copyBtn.innerHTML = '<i class="far fa-copy"></i> Copy Plan'; }, 1500);
            });
            actionsDiv.appendChild(copyBtn);

            // Trade
            const tradeBtn = document.createElement('button');
            tradeBtn.textContent = '🚀 Trade Now';
            if (r.can_trade) {
                tradeBtn.className = 'trade-btn';
                tradeBtn.addEventListener('click', () => tradeNow(r.pair, r.signal, r.price, r.tp, r.sl));
            } else {
                tradeBtn.className = 'trade-btn trade-btn-disabled';
                tradeBtn.disabled = true;
                tradeBtn.title = r.can_trade_reason || 'Trade not possible';
            }
            actionsDiv.appendChild(tradeBtn);

            card.appendChild(actionsDiv);
            actionableDiv.appendChild(card);
        });
    }

    // ---------- Trade Execution ----------
    async function tradeNow(pair, signal, price, tp, sl) {
        const volume = parseFloat(volumeInput?.value) || 0.01;
        const payload = {
            pairs: pair + '=X',
            interval: intervalSelect.value,
            atr_period: parseInt(atrSlider.value),
            risk_mult: parseFloat(riskSlider.value),
            volume: volume
        };
        try {
            const resp = await fetch('/api/autotrade', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const result = await resp.json();
            if (result[0]?.trade?.success) {
                alert(`✅ Trade executed: ${result[0].trade.message}`);
            } else {
                alert(`❌ Trade failed: ${result[0]?.trade?.error || 'Unknown error'}`);
            }
        } catch (err) {
            alert('❌ Trade failed: ' + err.message);
        }
    }

    // ---------- Copy plan ----------
    function copyTradePlan(r) {
        const text = `${r.signal} ${r.pair} at ${r.price}\nTP: ${r.tp} (3:1 R:R)\nSL: ${r.sl}\nATR used: ${r.atr}`;
        navigator.clipboard.writeText(text);
    }

    // ---------- Charts ----------
    function renderCharts(results) {
        if (!results.length) return;
        const title = document.createElement('h2');
        title.innerHTML = '<i class="fas fa-chart-line"></i> Price Action + Signal';
        chartsDiv.appendChild(title);

        results.forEach(r => {
            if (!r.chart || !r.chart.dates.length) return;
            const expander = document.createElement('div');
            expander.className = 'expandable';
            const header = document.createElement('div');
            header.className = 'expandable-header';
            header.innerHTML = `${r.pair} – ${r.signal} (${r.confidence}) <span><i class="fas fa-chevron-down"></i></span>`;
            header.addEventListener('click', () => {
                const content = expander.querySelector('.expandable-content');
                content.classList.toggle('show');
                const icon = header.querySelector('i');
                icon.className = content.classList.contains('show') ? 'fas fa-chevron-up' : 'fas fa-chevron-down';
            });
            const content = document.createElement('div');
            content.className = 'expandable-content';
            const chartDiv = document.createElement('div');
            chartDiv.className = 'chart-container';
            content.appendChild(chartDiv);
            expander.appendChild(header);
            expander.appendChild(content);
            chartsDiv.appendChild(expander);

            const trace = {
                x: r.chart.dates,
                y: r.chart.prices,
                mode: 'lines',
                name: r.pair,
                line: { color: '#3b82f6', width: 2 }
            };
            const layout = {
                title: '',
                paper_bgcolor: '#1a1f2b',
                plot_bgcolor: '#13161f',
                font: { color: '#edf2f7' },
                xaxis: { gridcolor: '#2a2f3c' },
                yaxis: { gridcolor: '#2a2f3c' },
                margin: { t: 20, l: 50, r: 30, b: 30 }
            };
            const data = [trace];
            if (r.chart.signal_point) {
                data.push({
                    x: [r.chart.signal_point.date],
                    y: [r.chart.signal_point.price],
                    mode: 'markers+text',
                    text: [r.chart.signal_point.signal],
                    textposition: 'top center',
                    marker: { size: 12, color: r.chart.signal_point.signal === 'BUY' ? '#10b981' : '#ef4444' },
                    name: 'Signal'
                });
            }
            Plotly.newPlot(chartDiv, data, layout, { responsive: true });
        });
    }

    window.tradeNow = tradeNow;
});