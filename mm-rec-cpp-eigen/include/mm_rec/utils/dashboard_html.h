#pragma once

namespace mm_rec {
namespace ui {

constexpr char DASHBOARD_HTML[] = R"HTML(
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MM-Rec Dashboard</title>
    <script src="https://unpkg.com/navigo@8.11.1/lib/navigo.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        :root {
            --bg: #0f172a; --sidebar: #1e293b; --card: #1e293b;
            --text: #f8fafc; --text-muted: #94a3b8; --accent: #38bdf8;
            --success: #22c55e; --danger: #ef4444; --border: #334155;
        }
        body { font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); height: 100vh; overflow: hidden; }
        
        /* Sidebar */
        .sidebar { position: fixed; left: 0; top: 0; width: 240px; height: 100vh; background: var(--sidebar); display: flex; flex-direction: column; border-right: 1px solid var(--border); z-index: 100; }
        .sidebar-header { padding: 20px; border-bottom: 1px solid var(--border); }
        .sidebar-header h1 { font-size: 1.25rem; background: linear-gradient(135deg, var(--accent), #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .sidebar-nav { flex: 1; padding: 20px 0; }
        .nav-item { display: flex; align-items: center; padding: 12px 20px; color: var(--text-muted); text-decoration: none; transition: 0.2s; cursor: pointer; border-left: 3px solid transparent; }
        .nav-item:hover { background: rgba(56, 189, 248, 0.1); color: var(--accent); }
        .nav-item.active { background: rgba(56, 189, 248, 0.15); color: var(--accent); border-left-color: var(--accent); }
        .nav-item i { width: 20px; margin-right: 12px; }
        
        /* Main Content */
        .main { margin-left: 240px; flex: 1; display: flex; flex-direction: column; overflow: hidden; }
        .top-bar { background: var(--sidebar); padding: 16px 24px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; }
        .top-bar-title { font-size: 1.25rem; font-weight: 600; }
        .btn { padding: 8px 16px; border: none; border-radius: 6px; cursor: pointer; transition: 0.2s; }
        .btn-primary { background: var(--accent); color: var(--bg); }
        .btn-danger { background: var(--danger); color: white; }
        .btn:hover { transform: translateY(-1px); }
        
        /* Views */
        .view { flex: 1; overflow-y: auto; padding: 24px; display: none; }
        .view.active { display: block; }
        .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 24px; }
        .card { background: var(--card); padding: 20px; border-radius: 12px; border: 1px solid var(--border); }
        .card-title { font-size: 0.875rem; color: var(--text-muted); text-transform: uppercase; margin-bottom: 8px; }
        .card-value { font-size: 2rem; font-weight: 700; }
        
        /* Table */
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid var(--border); }
        th { font-size: 0.875rem; color: var(--text-muted); font-weight: 600; }
        .badge { padding: 4px 12px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }
        .badge-success { background: rgba(34, 197, 94, 0.2); color: var(--success); }
        .badge-muted { background: rgba(148, 163, 184, 0.2); color: var(--text-muted); }
    </style>
</head>
<body>
    <div id="app"></div>
    
    <script>
    // Components
    function Sidebar() {
        return `
            <div class="sidebar">
                <div class="sidebar-header"><h1>MM-Rec</h1></div>
                <nav class="sidebar-nav">
                    <a href="#/" class="nav-item" data-view="training">
                        <i class="fas fa-chart-line"></i><span>Training</span>
                    </a>
                    <a href="#/runs" class="nav-item" data-view="runs">
                        <i class="fas fa-list"></i><span>Runs</span>
                    </a>
                    <a href="#/hardware" class="nav-item" data-view="hardware">
                        <i class="fas fa-microchip"></i><span>Hardware</span>
                    </a>
                </nav>
            </div>
        `;
    }
    
    function TopBar() {
        return `
            <div class="top-bar">
                <div class="top-bar-title">Training Dashboard</div>
                <div>
                    <button class="btn btn-danger" onclick="fetch('/api/stop')">
                        <i class="fas fa-stop"></i> Stop
                    </button>
                </div>
            </div>
        `;
    }
    
    function TrainingView() {
        return `
            <div id="training-view" class="view active">
                <div class="cards">
                    <div class="card"><div class="card-title">Loss</div><div class="card-value" id="loss">--</div></div>
                    <div class="card"><div class="card-title">Step</div><div class="card-value" id="step">--</div></div>
                    <div class="card"><div class="card-title">Speed</div><div class="card-value" id="speed">--</div></div>
                    <div class="card"><div class="card-title">GPU Ratio</div><div class="card-value" id="gpu_ratio">--</div></div>
                </div>
                <div class="card"><canvas id="chart"></canvas></div>
            </div>
        `;
    }
    
    function RunsView() {
        return `
            <div id="runs-view" class="view">
                <div class="card">
                    <h2>Training Runs</h2>
                    <table><thead><tr>
                        <th>Name</th><th>Status</th><th>Epoch</th><th>Loss</th><th>Size</th>
                    </tr></thead>
                    <tbody id="runs-table"><tr><td colspan="5">Loading...</td></tr></tbody></table>
                </div>
            </div>
        `;
    }
    
    function HardwareView() {
        return `
            <div id="hardware-view" class="view">
                <div class="card">
                    <h2>Hardware</h2>
                    <table>
                        <tr><td>Processor</td><td id="hw-cpu">Loading...</td></tr>
                        <tr><td>Architecture</td><td id="hw-arch">-</td></tr>
                        <tr><td>Memory</td><td id="hw-ram">-</td></tr>
                        <tr><td>Compute</td><td id="hw-compute">-</td></tr>
                    </table>
                </div>
            </div>
        `;
    }
    
    // Render
    document.getElementById('app').innerHTML = `
        ${Sidebar()}
        <div class="main">
            ${TopBar()}
            ${TrainingView()}
            ${RunsView()}
            ${HardwareView()}
        </div>
    `;
    
    // Chart
    const chart = new Chart(document.getElementById('chart'), {
        type: 'line',
        data: { labels: [], datasets: [{ label: 'Loss', data: [], borderColor: '#38bdf8', tension: 0.4 }] },
        options: { responsive: true, maintainAspectRatio: false }
    });
    
    // Router
    const router = new Navigo('/', { hash: true });
    
    function showView(view) {
        document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
        document.getElementById(view + '-view').classList.add('active');
        document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
        document.querySelector(`[data-view="${view}"]`).classList.add('active');
    }
    
    router
        .on('/', () => showView('training'))
        .on('/runs', async () => {
            showView('runs');
            const r = await fetch('/api/runs');
            const runs = await r.json();
            document.getElementById('runs-table').innerHTML = runs.map(x => 
                `<tr><td>${x.name}</td><td><span class="badge badge-${x.status === 'RUNNING' ? 'success' : 'muted'}">${x.status}</span></td>
                <td>${x.epoch}</td><td>${x.loss > 0 ? x.loss.toFixed(3) : '-'}</td><td>${x.size_mb} MB</td></tr>`
            ).join('');
        })
        .on('/hardware', async () => {
            showView('hardware');
            const h = await fetch('/api/hardware');
            const hw = await h.json();
            document.getElementById('hw-cpu').textContent = hw.cpu_model;
            document.getElementById('hw-arch').textContent = hw.arch;
            document.getElementById('hw-ram').textContent = (hw.mem_total_mb / 1024).toFixed(1) + ' GB';
            document.getElementById('hw-compute').textContent = hw.compute_device;
        })
        .resolve();
    
    // Stats polling
    async function fetchStats() {
        try {
            const r = await fetch('/api/stats');
            const d = await r.json();
            document.getElementById('loss').textContent = d.loss?.toFixed(4) || '--';
            document.getElementById('step').textContent = d.step || '--';
            document.getElementById('speed').textContent = d.speed?.toFixed(1) || '--';
            document.getElementById('gpu_ratio').textContent = d.gpu_ratio?.toFixed(2) || '--';
            if (d.history?.length > 0) {
                chart.data.labels = d.history.map((_, i) => i);
                chart.data.datasets[0].data = d.history;
                chart.update();
            }
        } catch (e) {}
    }
    setInterval(fetchStats, 1000);
    fetchStats();
    </script>
</body>
</html>
)HTML";

} // namespace ui
} // namespace mm_rec
