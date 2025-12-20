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
                <th>Name</th><th>Status</th><th>Epoch</th><th>Loss</th><th>Size</th><th>Actions</th>
            </tr></thead>
            <tbody id="runs-table"><tr><td colspan="6">Loading...</td></tr></tbody></table>
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
        document.getElementById('runs-table').innerHTML = runs.map(x => {
            let actions = '';
            // Only show actions if status is known
            if (x.status === 'RUNNING') {
                actions = `<button onclick="stopRun('${x.name}')" class="btn btn-sm btn-danger" title="Stop"><i class="fas fa-stop"></i></button>`;
            } else {
                actions = `
                    <button onclick="resumeRun('${x.name}')" class="btn btn-sm btn-primary" title="Resume"><i class="fas fa-play"></i></button>
                    <button onclick="deleteRun('${x.name}')" class="btn btn-sm btn-danger" title="Delete"><i class="fas fa-trash"></i></button>
                `;
            }
            return `<tr>
                <td>${x.name}</td>
                <td><span class="badge badge-${x.status === 'RUNNING' ? 'success' : 'muted'}">${x.status}</span></td>
                <td>${x.epoch}</td>
                <td>${x.loss > 0 ? x.loss.toFixed(3) : '-'}</td>
                <td>${x.total_size_mb || x.size_mb} MB</td>
                <td>${actions}</td>
            </tr>`;
        }).join('');
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
    } catch (e) { }
}

// Run management functions
async function resumeRun(name) {
    if (confirm(`Resume training run: ${name}?`)) {
        await fetch(`/api/runs/resume?name=${name}`, { method: 'POST' });
        setTimeout(() => router.navigate('/runs'), 500);
    }
}

async function stopRun(name) {
    if (confirm(`Stop active training run?`)) {
        await fetch(`/api/runs/stop`, { method: 'POST' });
        setTimeout(() => router.navigate('/runs'), 500);
    }
}

async function deleteRun(name) {
    if (confirm(`Delete training run: ${name}? This cannot be undone!`)) {
        await fetch(`/api/runs/delete?name=${name}`, { method: 'DELETE' });
        setTimeout(() => router.navigate('/runs'), 500);
    }
}

setInterval(fetchStats, 1000);
fetchStats();
