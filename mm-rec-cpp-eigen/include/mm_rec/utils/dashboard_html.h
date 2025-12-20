#pragma once

namespace mm_rec {
namespace ui {

constexpr char DASHBOARD_HTML[] = R"HTML(
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MM-Rec Training Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root { --bg: #0f172a; --card: #1e293b; --text: #f8fafc; --accent: #38bdf8; --danger: #ef4444; }
        body { font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .card { background: var(--card); padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
        .full-width { grid-column: span 2; }
        h2 { margin-top: 0; color: var(--accent); font-size: 1.2rem; border-bottom: 1px solid #334155; padding-bottom: 10px; }
        .stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; }
        .stat-item { text-align: center; }
        .stat-val { font-size: 1.8rem; font-weight: bold; }
        .stat-label { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; }
        
        .btn { border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer; font-weight: bold; transition: 0.2s; }
        .btn-danger { background: var(--danger); color: white; }
        .btn-danger:hover { background: #dc2626; }
        
        canvas { max-height: 400px; }
        
        /* Modal */
        .modal { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; overflow: auto; background-color: rgba(0,0,0,0.8); backdrop-filter: blur(5px); }
        .modal-content { background-color: #1e293b; margin: 10% auto; padding: 0; border: 1px solid #334155; width: 500px; border-radius: 12px; box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5); }
        .modal-header { padding: 20px; border-bottom: 1px solid #334155; display: flex; justify-content: space-between; align-items: center; }
        .modal-body { padding: 20px; }
        .close { color: #aaa; float: right; font-size: 28px; font-weight: bold; cursor: pointer; }
        .close:hover { color: #fff; }
        .hw-row { padding: 10px 0; border-bottom: 1px solid #334155; display: flex; justify-content: space-between; }
        .hw-row:last-child { border-bottom: none; }
        .hw-row span { color: #94a3b8; }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="card full-width" style="display: flex; justify-content: space-between; align-items: center;">
            <div style="display: flex; align-items: center; gap: 15px;">
                <div style="width: 12px; height: 12px; background: #22c55e; border-radius: 50%; box-shadow: 0 0 10px #22c55e;"></div>
                <h1 style="margin: 0; font-size: 1.5rem;">MM-Rec Training Monitor</h1>
            </div>
            <div style="display: flex; gap: 10px;">
                <button id="histBtn" class="btn" style="background: #8b5cf6;" onclick="loadHistory()">LOAD HISTORY</button>
                <button class="btn" style="background: #3b82f6;" onclick="openHwModal()">HARDWARE</button>
                <button class="btn stop-btn" onclick="stopTraining()">STOP TRAINING</button>
            </div>
        </div>
    </div>

    <div class="grid-container">
        <!-- Hardware Modal -->
        <div id="hwModal" class="modal">
            <div class="modal-content">
                <div class="modal-header">
                    <h2>Hardware Specs</h2>
                    <span class="close" onclick="closeHwModal()">&times;</span>
                </div>
                <div class="modal-body">
                    <div class="hw-row"><strong>Processor:</strong> <span id="hw-cpu">Loading...</span></div>
                    <div class="hw-row"><strong>Architecture:</strong> <span id="hw-arch">-</span></div>
                    <div class="hw-row"><strong>Cores:</strong> <span id="hw-cores">-</span></div>
                    <div class="hw-row"><strong>SIMD:</strong> <span id="hw-simd">-</span></div>
                    <div class="hw-row"><strong>Memory (Total):</strong> <span id="hw-ram">-</span></div>
                    <div class="hw-row"><strong>Compute Mode:</strong> <span id="hw-compute">-</span></div>
                </div>
            </div>
        </div>
        
        <!-- Main Status (Full Width) -->
        <div class="card full-width">
            <div class="stats-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 15px;">
                <div class="stat-item">
                    <div class="stat-label">Loss</div>
                    <div class="stat-value" id="loss">--</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Epoch</div>
                    <div class="stat-value" id="epoch">--</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Step</div>
                    <div class="stat-value" id="step">--</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">ETA</div>
                    <div class="stat-value" id="eta">Calc...</div>
                </div>
                 <div class="stat-item">
                    <div class="stat-label">Speed</div>
                    <div class="stat-value" id="speed">--</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">LR</div>
                    <div class="stat-value" id="lr">--</div>
                </div>
                 <div class="stat-item">
                    <div class="stat-label">RAM</div>
                    <div class="stat-value" id="mem">--</div>
                </div>
                 <div class="stat-item">
                    <div class="stat-label">Device</div>
                    <div class="stat-value">HYBRID</div>
                </div>
            </div>
        </div>

        <!-- Charts (2 Columns) -->
        <div class="card">
            <h2>Loss History (Main)</h2>
            <div style="height: 250px;"><canvas id="lossChart"></canvas></div>
        </div>
        
        <div class="card">
            <h2>Health (Grads & LR)</h2>
            <div style="height: 250px;"><canvas id="healthChart"></canvas></div>
        </div>

        <div class="card full-width"> <!-- Perf can be full width or half -->
            <h2>System Performance (Stall)</h2>
            <div style="height: 200px;"><canvas id="perfChart"></canvas></div>
        </div>
        
        <!-- Hybrid Metrics Card -->
        <div class="card full-width">
            <h2>Hybrid Execution (CPU <-> GPU)</h2>
            <div style="display: flex; align-items: center; gap: 20px;">
                <div style="flex: 1;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span>Workload Distribution</span>
                        <span><span id="gpu-ratio-val">50</span>% GPU</span>
                    </div>
                    <div style="background: #334155; height: 20px; border-radius: 10px; overflow: hidden; position: relative;">
                        <!-- CPU portion (Left, Blue) -->
                        <div style="position: absolute; left: 0; top: 0; bottom: 0; width: 100%; background: #3b82f6;"></div>
                        <!-- GPU portion (Right, Purple overlay) -->
                        <div id="gpu-bar" style="position: absolute; left: 0; top: 0; bottom: 0; width: 50%; background: #a855f7; transition: width 0.5s;"></div>
                        <!-- Text Overlay -->
                        <div style="position: absolute; width: 100%; text-align: center; font-size: 12px; line-height: 20px; font-weight: bold; color: white; mix-blend-mode: difference;">
                            CPU (Left) vs GPU (Right)
                        </div>
                    </div>
                </div>
                
                <div style="flex: 0 0 150px; text-align: center;">
                    <div class="stat-label">Sync Delta</div>
                    <div class="stat-value" id="sync-delta" style="font-size: 1.5rem;">0.0 ms</div>
                    <div style="font-size: 0.8rem; color: #94a3b8;">(Lower is better)</div>
                </div>
            </div>
        </div>
    </div>

    <style>
        .grid-container { max-width: 1200px; margin: 20px auto 0 auto; display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .grid-container .full-width { grid-column: 1 / -1; }
        .stat-value { font-size: 1.8rem; font-weight: bold; }
        .stat-label { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; }
        @media (max-width: 1000px) { .grid-container { grid-template-columns: 1fr; } }
    </style>

    <script>
        // Custom Plugin: Zone Backgrounds
        const zonePlugin = {
            id: 'zonePlugin',
            beforeDraw: (chart) => {
                if (chart.config.options.plugins.zones) {
                    const ctx = chart.ctx;
                    const xAxis = chart.scales.x;
                    
                    chart.config.options.plugins.zones.forEach(zone => {
                        // Support custom axis (e.g., 'y1' for Health Chart, 'y' for Loss Chart)
                        const axisId = zone.axisID || 'y';
                        const yAxis = chart.scales[axisId];
                        
                        // Safety: Skip if axis doesn't exist (e.g., wrong config or not ready)
                        if (!yAxis) return;

                        const yTop = yAxis.getPixelForValue(zone.max);
                        const yBottom = yAxis.getPixelForValue(zone.min);
                        
                        ctx.save();
                        ctx.fillStyle = zone.color;
                        ctx.fillRect(xAxis.left, yTop, xAxis.width, yBottom - yTop);
                        ctx.restore();
                        
                        // Label
                        if (zone.label) {
                            ctx.fillStyle = zone.labelColor || '#64748b';
                            ctx.font = '10px sans-serif';
                            ctx.fillText(zone.label, xAxis.right - 60, yTop + 12);
                        }
                    });
                }
            }
        };
        Chart.register(zonePlugin);

        // --- 1. Loss Chart ---
        const ctxLoss = document.getElementById('lossChart').getContext('2d');
        const lossChart = new Chart(ctxLoss, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { label: 'Raw Loss', data: [], borderColor: 'rgba(56, 189, 248, 0.3)', backgroundColor: 'rgba(56,189,248,0.05)', borderWidth: 1, tension: 0.3, fill: true, pointRadius: 0 },
                    { label: 'Trend (EMA)', data: [], borderColor: '#facc15', borderWidth: 3, tension: 0.4, fill: false, pointRadius: 0, shadowBlur: 10, shadowColor: 'black' }
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false, animation: false,
                plugins: { 
                    legend: { labels: { color: '#94a3b8' } },
                    zones: [
                        { min: 0, max: 1.0, color: 'rgba(34, 197, 94, 0.1)', label: 'Mastery Zone' }, // Green
                        { min: 10, max: 20, color: 'rgba(239, 68, 68, 0.1)', label: 'Chaos Zone' }    // Red
                    ]
                },
                scales: { x: { display: false }, y: { grid: { color: '#334155' } } }
            }
        });

        // --- 2. Health Chart (Grad Norm & LR) ---
        const ctxHealth = document.getElementById('healthChart').getContext('2d');
        const healthChart = new Chart(ctxHealth, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { 
                        label: 'Gradient Norm', 
                        data: [], 
                        borderColor: '#f43f5e', // Rose-500
                        backgroundColor: 'rgba(244, 63, 94, 0.1)',
                        borderWidth: 1, 
                        yAxisID: 'y1',
                        pointRadius: 0,
                        fill: true
                    },
                    { 
                        label: 'Learning Rate', 
                        data: [], 
                        borderColor: '#22c55e', // Green-500
                        borderWidth: 2, 
                        yAxisID: 'y2',
                        pointRadius: 0,
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false, animation: false,
                plugins: { 
                    legend: { labels: { color: '#94a3b8' } },
                    zones: [
                        { min: 1.0, max: 100.0, color: 'rgba(239, 68, 68, 0.1)', label: 'Clipping Zone (Instability)', axisID: 'y1' } 
                    ]
                },
                scales: { 
                    x: { display: false }, 
                    y1: { type: 'linear', display: true, position: 'left', grid: { color: '#334155' }, title: {display: true, text: 'Grad Norm'} },
                    y2: { type: 'linear', display: true, position: 'right', grid: { drawOnChartArea: false }, title: {display: true, text: 'LR'} }
                }
            }
        });

        // --- 3. Performance Chart (Data Stall) ---
        const ctxPerf = document.getElementById('perfChart').getContext('2d');
        const perfChart = new Chart(ctxPerf, {
            type: 'bar', // Bar chart to show distinct stalls
            data: {
                labels: [],
                datasets: [
                    { 
                        label: 'Data Stall (ms)', 
                        data: [], 
                        backgroundColor: (ctx) => {
                            const val = ctx.raw;
                            return val > 20 ? '#ef4444' : '#3b82f6'; // Red if > 20ms, Blue otherwise
                        },
                        barThickness: 2
                    }
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false, animation: false,
                plugins: { legend: { display: false } },
                scales: { 
                    x: { display: false }, 
                    y: { grid: { color: '#334155' }, title: {display: true, text: 'Wait Time (ms)'}, min: 0 }
                }
            }
        });

        // Modal Logic
        const modal = document.getElementById("hwModal");
        async function openHwModal() {
            modal.style.display = "block";
            try {
                const res = await fetch('/api/hardware');
                const hw = await res.json();
                document.getElementById('hw-cpu').innerText = hw.cpu_model;
                document.getElementById('hw-arch').innerText = hw.arch;
                document.getElementById('hw-cores').innerText = hw.cores_logical + " (Log)";
                document.getElementById('hw-simd').innerText = hw.simd;
                document.getElementById('hw-ram').innerText = (hw.mem_total_mb / 1024).toFixed(1) + " GB";
                document.getElementById('hw-compute').innerText = hw.compute_device;
            } catch(e) { console.error(e); }
        }
        function closeHwModal() { modal.style.display = "none"; }
        window.onclick = function(event) { if (event.target == modal) closeHwModal(); }

        async function fetchStats() {
            try {
                const res = await fetch('/api/stats');
                const data = await res.json();
                
                // Safety Checks & formatting
                const safeFloat = (val, prec) => val != null ? val.toFixed(prec) : '--';
                const safeInt = (val) => val != null ? val : '--';
                
                document.getElementById('loss').innerText = safeFloat(data.loss, 4);
                
                // Update Hybrid Metrics
                if (data.gpu_ratio != null) {
                    const pct = (data.gpu_ratio * 100).toFixed(1);
                    document.getElementById('gpu-bar').style.width = pct + "%";
                    document.getElementById('gpu-ratio-val').innerText = pct;
                }
                if (data.sync_delta != null) {
                    const delta = data.sync_delta.toFixed(2);
                    const el = document.getElementById('sync-delta');
                    el.innerText = delta + " ms";
                    // Color code
                    if (Math.abs(data.sync_delta) < 2.0) el.style.color = "#22c55e"; // Green
                    else el.style.color = "#ef4444"; // Red
                }

                document.getElementById('epoch').innerText = safeInt(data.epoch);
                document.getElementById('step').innerText = safeInt(data.step);
                document.getElementById('lr').innerText = safeFloat(data.lr, 6);
                document.getElementById('speed').innerText = safeFloat(data.speed, 1) + " tok/s";
                document.getElementById('mem').innerText = safeInt(data.mem) + " MB";
                
                // ETA Calculation
                const totalSteps = data.total_steps || 0;
                const currentStep = data.step || 0;
                const speed = data.speed || 0;
                
                if (totalSteps > 0 && speed > 0 && currentStep < totalSteps) {
                    const remainingSteps = totalSteps - currentStep;
                    const remainingSeconds = remainingSteps / speed; // Steps / (Tokens/sec)?? Wait, speed is Tokens/sec?
                    // Speed is speed_. usually steps/sec or tokens/sec. 
                    // Let's assume it's tokens/sec and use an arbitrary multiplier if needed, 
                    // BUT Trainer::current_speed_ is usually passed updates.
                    // Actually previous code said "1.04 it/s". So speed IS steps/sec (iterations/sec).
                    // So remainingSeconds = remainingSteps / speed.
                    
                    if (remainingSeconds < 60) document.getElementById('eta').innerText = remainingSeconds.toFixed(0) + "s";
                    else if (remainingSeconds < 3600) document.getElementById('eta').innerText = (remainingSeconds/60).toFixed(0) + "m";
                    else document.getElementById('eta').innerText = (remainingSeconds/3600).toFixed(1) + "h";
                } else {
                    document.getElementById('eta').innerText = "--";
                }

                // Update Loss Chart
                if (!showingHistory && data.history && data.history.length > 0) {
                    lossChart.data.labels = data.history.map((_, i) => i);
                    lossChart.data.datasets[0].data = data.history;
                    if (data.avg_history) lossChart.data.datasets[1].data = data.avg_history;
                    lossChart.update();
                }
                
                // Update Health Chart
                if (data.grad_norm_history && data.grad_norm_history.length > 0) {
                    healthChart.data.labels = data.grad_norm_history.map((_, i) => i);
                    healthChart.data.datasets[0].data = data.grad_norm_history;
                    if (data.lr_history) healthChart.data.datasets[1].data = data.lr_history;
                    healthChart.update();
                }

                // Update Perf Chart
                if (data.data_stall_history && data.data_stall_history.length > 0) {
                    perfChart.data.labels = data.data_stall_history.map((_, i) => i);
                    perfChart.data.datasets[0].data = data.data_stall_history;
                    perfChart.update();
                }
                
            } catch (e) {
                console.error("Connection lost", e);
            }
        }

        let showingHistory = false;
        async function loadHistory() {
            const btn = document.getElementById('histBtn');
            if (showingHistory) {
                // Swith back to Live
                showingHistory = false;
                btn.innerText = "LOAD HISTORY";
                btn.style.background = "#8b5cf6";
                return;
            }

            btn.innerText = "Loading...";
            try {
                const res = await fetch('/api/history');
                const data = await res.json();
                if (data.loss_history && data.loss_history.length > 0) {
                    lossChart.data.labels = data.loss_history.map((_, i) => i);
                    lossChart.data.datasets[0].data = data.loss_history;
                    // Optional: Calculate EMA locally or just hide it
                    lossChart.data.datasets[1].data = []; 
                    lossChart.update();
                    
                    showingHistory = true;
                    btn.innerText = "LIVE VIEW";
                    btn.style.background = "#22c55e";
                } else {
                    alert("No history found yet.");
                    btn.innerText = "LOAD HISTORY";
                }
            } catch(e) { 
                console.error(e); 
                btn.innerText = "ERROR"; 
            }
        }

        async function stopTraining() {
            if (confirm('Are you sure you want to stop training?')) {
                const btn = document.querySelector('button.stop-btn'); // Assuming class or ID
                if (btn) {
                    btn.disabled = true;
                    btn.innerText = "Stopping... ⏳";
                    btn.style.backgroundColor = "#666";
                }
                
                try {
                    await fetch('/api/stop', { method: 'POST' });
                    // alert('Stop signal sent!'); // Remove blocking alert
                } catch (e) {
                    alert('Failed to send stop signal');
                    if (btn) btn.disabled = false;
                }
            }
        }

        // Poll every 1s
        setInterval(fetchStats, 1000);
        fetchStats();
    </script>
</body>
</html>
)HTML";

} // namespace ui
} // namespace mm_rec
