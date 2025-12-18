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
            <button class="btn btn-danger" onclick="stopTraining()">STOP TRAINING</button>
        </div>

        <!-- Stats -->
        <div class="card full-width">
            <div class="stat-grid">
                <div class="stat-item">
                    <div class="stat-val" id="loss">0.000</div>
                    <div class="stat-label">Current Loss</div>
                </div>
                <div class="stat-item">
                    <div class="stat-val" id="epoch">0</div>
                    <div class="stat-label">Epoch</div>
                </div>
                <div class="stat-item">
                    <div class="stat-val" id="lr">0.0000</div>
                    <div class="stat-label">Learning Rate</div>
                </div>
                <div class="stat-item">
                    <div class="stat-val" id="speed">0 t/s</div>
                    <div class="stat-label">Throughput</div>
                </div>
            </div>
        </div>

        <!-- Charts -->
        <div class="card full-width">
            <h2>Loss History</h2>
            <canvas id="lossChart"></canvas>
        </div>
    </div>

    <script>
        const ctx = document.getElementById('lossChart').getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Training Loss',
                    data: [],
                    borderColor: '#38bdf8',
                    backgroundColor: 'rgba(56, 189, 248, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true,
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { display: false },
                    y: { grid: { color: '#334155' } }
                }
            }
        });

        async function fetchStats() {
            try {
                const res = await fetch('/api/stats');
                const data = await res.json();
                
                document.getElementById('loss').innerText = data.loss.toFixed(4);
                document.getElementById('epoch').innerText = data.epoch;
                document.getElementById('lr').innerText = data.lr.toFixed(6);
                document.getElementById('speed').innerText = data.speed.toFixed(0) + " t/s";

                // Update Chart
                if (data.history && data.history.length > 0) {
                    chart.data.labels = data.history.map((_, i) => i);
                    chart.data.datasets[0].data = data.history;
                    chart.update();
                }
            } catch (e) {
                console.error("Connection lost", e);
            }
        }

        async function stopTraining() {
            if (confirm('Are you sure you want to stop training?')) {
                await fetch('/api/stop', { method: 'POST' });
                alert('Stop signal sent!');
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
