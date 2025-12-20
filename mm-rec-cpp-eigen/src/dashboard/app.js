const app = {
    // State
    chart: null,
    lossHistory: [],
    pollInterval: null,
    currentTab: 'overview',

    init: function () {
        console.log("Initializing Dashboard...");
        this.initChart();
        this.startPolling();
        this.updateRuns(); // Initial fetch
        this.fetchModels(); // Initial fetch for Models tab
    },

    startPolling: function () {
        if (this.pollInterval) clearInterval(this.pollInterval);
        this.pollInterval = setInterval(() => {
            this.fetchStats();
            this.updateRuns(); // Refresh runs list periodically
        }, 1000);
    },

    // --- TABS ---
    switchTab: function (tabName) {
        this.currentTab = tabName;

        // Buttons
        document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
        const activeBtn = document.querySelector(`.tab-btn[onclick*="${tabName}"]`);
        if (activeBtn) activeBtn.classList.add('active');

        // Content
        document.querySelectorAll('.tab-content').forEach(div => div.style.display = 'none');
        document.getElementById(`tab-${tabName}`).style.display = 'block';

        // Lazy Load
        if (tabName === 'configs') this.fetchConfigs();
        if (tabName === 'datasets') this.fetchDatasets();
        if (tabName === 'models') this.fetchModels();
    },

    // --- STATS & CHARTS ---
    initChart: function () {
        const ctx = document.getElementById('loss-chart').getContext('2d');
        this.chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Training Loss',
                    data: [],
                    borderColor: '#3498db',
                    tension: 0.1,
                    borderWidth: 2,
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                scales: {
                    x: { display: false },
                    y: { grid: { color: '#333' } }
                }
            }
        });
    },

    fetchStats: function () {
        fetch('/api/stats')
            .then(r => r.json())
            .then(data => {
                // Update Values
                document.getElementById('loss-value').textContent = data.loss.toFixed(6);
                document.getElementById('step-value').textContent = data.step;
                document.getElementById('lr-value').textContent = data.lr.toExponential(2);
                document.getElementById('speed-value').textContent = data.speed.toFixed(1) + " tok/s";
                document.getElementById('mem-value').textContent = data.mem + " MB";
                document.getElementById('sync-value').textContent = (data.sync_delta * 1000).toFixed(2) + " ms";
                document.getElementById('server-status').textContent = "Connected (Step " + data.step + ")";
                document.getElementById('server-status').style.color = "#2ecc71";

                // Update Chart
                if (data.history && data.history.length > 0) {
                    // Simple shift logic or full replace
                    this.chart.data.labels = data.history.map((_, i) => i);
                    this.chart.data.datasets[0].data = data.history;
                    this.chart.update();
                }
            })
            .catch(e => {
                document.getElementById('server-status').textContent = "Disconnected";
                document.getElementById('server-status').style.color = "#e74c3c";
            });

        // Hardware (Separate call or less frequent)
        fetch('/api/hardware').then(r => r.json()).then(hw => {
            const container = document.getElementById('hardware-info');
            container.innerHTML = `
                <div><strong>GPU:</strong> ${hw.compute_device}</div>
                <div><strong>VRAM:</strong> ${hw.mem_total_mb} MB</div>
                <div><strong>CPU:</strong> ${hw.cpu_model}</div>
                <div><strong>SIMD:</strong> ${hw.simd}</div>
            `;
        }).catch(() => { });
    },

    // --- RUNS MANAGEMENT ---
    updateRuns: function () {
        if (this.currentTab !== 'overview') return; // Optimize

        fetch('/api/runs')
            .then(r => r.json())
            .then(runs => {
                const tbody = document.getElementById('runs-body');
                tbody.innerHTML = '';
                runs.forEach(run => {
                    const tr = document.createElement('tr');

                    // Status Badge Color
                    let statusColor = '#888';
                    if (run.status === 'RUNNING') statusColor = '#2ecc71';
                    else if (run.status === 'FAILED') statusColor = '#e74c3c';
                    else if (run.status === 'COMPLETED') statusColor = '#f1c40f';

                    // Actions Logic
                    let actions = '';
                    if (run.status === 'RUNNING') {
                        actions = `<button class="btn-small btn-danger" onclick="app.stopRun('${run.name}')" title="Stop"><i class="fas fa-stop"></i></button>`;
                    } else {
                        // Resume
                        actions += `<button class="btn-small btn-success" onclick="app.resumeRun('${run.name}')" title="Resume"><i class="fas fa-play"></i></button>`;
                        // Delete
                        if (run.status !== 'RUNNING') {
                            actions += `<button class="btn-small btn-danger" onclick="app.deleteRun('${run.name}')" title="Delete"><i class="fas fa-trash"></i></button>`;
                        }
                    }

                    tr.innerHTML = `
                        <td>${run.name}</td>
                        <td><span style="color:${statusColor}">${run.status}</span></td>
                        <td>${run.epoch}</td>
                        <td>${run.loss.toFixed(4)}</td>
                        <td>${run.best_loss.toFixed(4)}</td>
                        <td>${run.size_mb}</td>
                        <td>${actions}</td>
                    `;
                    tbody.appendChild(tr);
                });
            });
    },

    stopRun: function (name) {
        if (!confirm("Stop training '" + name + "'?")) return;
        fetch('/api/runs/stop', { method: 'POST' }).then(() => this.updateRuns());
    },

    resumeRun: function (name) {
        fetch('/api/runs/resume?name=' + name, { method: 'POST' })
            .then(r => r.json())
            .then(d => {
                if (d.error) alert(d.error);
                this.updateRuns();
            });
    },

    deleteRun: function (name) {
        if (!confirm("Delete run '" + name + "' and all data?")) return;
        fetch('/api/runs/delete?name=' + name, { method: 'DELETE' })
            .then(r => r.json())
            .then(d => {
                if (d.error) alert(d.error);
                this.updateRuns();
            });
    },

    // --- NEW RUN MODAL ---
    openNewRunModal: function () {
        document.getElementById('modal-new-run').style.display = 'block';

        // Fetch Configs
        fetch('/api/configs').then(r => r.json()).then(configs => {
            const sel = document.getElementById('new-run-config');
            sel.innerHTML = '';
            configs.forEach(c => {
                // c is filename string? or object? implementation returned list of strings
                const opt = document.createElement('option');
                opt.value = c;
                opt.textContent = c;
                sel.appendChild(opt);
            });
        });

        // Fetch Datasets
        fetch('/api/datasets').then(r => r.json()).then(datasets => {
            const sel = document.getElementById('new-run-dataset');
            sel.innerHTML = '';
            datasets.forEach(d => {
                const opt = document.createElement('option');
                opt.value = d.name;
                opt.textContent = `${d.name} (${d.size_mb} MB)`;
                sel.appendChild(opt);
            });
        });
    },

    startNewRun: function () {
        const name = document.getElementById('new-run-name').value.trim();
        const config = document.getElementById('new-run-config').value;
        const dataset = document.getElementById('new-run-dataset').value;

        if (!config || !dataset) {
            alert("Please select config and dataset");
            return;
        }

        const payload = {
            run_name: name,
            config_file: config,
            data_file: dataset
        };

        fetch('/api/runs/start', {
            method: 'POST',
            body: JSON.stringify(payload),
            headers: { 'Content-Type': 'application/json' }
        })
            .then(r => r.json())
            .then(data => {
                if (data.error) alert("Error: " + data.error);
                else {
                    this.closeModal('modal-new-run');
                    this.switchTab('overview');
                    this.updateRuns();
                }
            });
    },

    // --- CONFIGS ---
    fetchConfigs: function () {
        fetch('/api/configs')
            .then(r => r.json())
            .then(items => {
                const list = document.getElementById('config-list');
                list.innerHTML = '';
                items.forEach(c => {
                    const li = document.createElement('li');
                    li.innerHTML = `<span><i class="fas fa-file-code"></i> ${c}</span>`;
                    list.appendChild(li);
                });
            });
    },

    openNewConfigModal: function () {
        document.getElementById('modal-new-config').style.display = 'block';
    },

    createConfig: function () {
        const filename = document.getElementById('new-config-filename').value.trim();
        const content = document.getElementById('new-config-content').value;

        fetch('/api/configs/create', {
            method: 'POST',
            body: JSON.stringify({ filename, content }),
            headers: { 'Content-Type': 'application/json' }
        }).then(r => r.json()).then(d => {
            if (d.error) alert(d.error);
            else {
                this.closeModal('modal-new-config');
                this.fetchConfigs();
            }
        });
    },

    // --- DATASETS ---
    fetchDatasets: function () {
        fetch('/api/datasets')
            .then(r => r.json())
            .then(items => {
                const list = document.getElementById('dataset-list');
                list.innerHTML = '';
                items.forEach(d => {
                    const li = document.createElement('li');
                    li.innerHTML = `<span><i class="fas fa-database"></i> ${d.name}</span> <span style="color:#888">${d.size_mb} MB</span>`;
                    list.appendChild(li);
                });
            });
    },

    uploadDataset: function () {
        const fileInput = document.getElementById('dataset-file-input');
        if (fileInput.files.length === 0) return;
        const file = fileInput.files[0];

        document.getElementById('upload-status').textContent = "Uploading...";

        fetch('/api/datasets/upload?name=' + encodeURIComponent(file.name), {
            method: 'PUT',
            body: file
        }).then(r => r.json()).then(d => {
            if (d.error) {
                document.getElementById('upload-status').textContent = "Error: " + d.error;
            } else {
                document.getElementById('upload-status').textContent = "Success!";
                this.fetchDatasets();
                fileInput.value = ''; // Reset
            }
        });
    },

    // --- MODELS & INFERENCE ---
    fetchModels: function () {
        fetch('/api/models').then(r => r.json()).then(items => {
            // Models List
            const list = document.getElementById('model-list');
            if (list) {
                list.innerHTML = '';
                items.forEach(m => {
                    const li = document.createElement('li');
                    li.innerHTML = `
                        <span><i class="fas fa-cube"></i> ${m.name}</span>
                        <div>
                            <span style="color:#888; margin-right:10px;">${m.size_mb} MB</span>
                            <a href="/api/models/download?name=${m.name}" target="_blank" class="btn-small btn-secondary"><i class="fas fa-download"></i></a>
                        </div>`;
                    list.appendChild(li);
                });
            }

            // Inference Select
            const sel = document.getElementById('inference-model-select');
            if (sel) {
                const current = sel.value;
                sel.innerHTML = '';
                items.forEach(m => {
                    const opt = document.createElement('option');
                    opt.value = m.name;
                    opt.textContent = m.name;
                    if (m.name === current) opt.selected = true;
                    sel.appendChild(opt);
                });
                if (!current && items.length > 0) sel.value = items[0].name;
            }
        });
    },

    runInference: function () {
        const model = document.getElementById('inference-model-select').value;
        const prompt = document.getElementById('inference-prompt').value;
        const resultBox = document.getElementById('inference-result');

        resultBox.textContent = "Generating...";

        fetch('/api/inference', {
            method: 'POST',
            body: JSON.stringify({ model, prompt }),
            headers: { 'Content-Type': 'application/json' }
        }).then(r => r.json()).then(d => {
            if (d.error) resultBox.textContent = "Error: " + d.error;
            else resultBox.textContent = d.text;
        });
    },

    // Utils
    closeModal: function (id) {
        document.getElementById(id).style.display = 'none';
    }
};

// Start
document.addEventListener('DOMContentLoaded', () => {
    app.init();
});
