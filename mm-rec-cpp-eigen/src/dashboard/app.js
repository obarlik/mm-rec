const app = {
    // State
    activeView: 'home',
    activeRun: null,
    activeDetailTab: 'monitor',
    chart: null,
    pollInterval: null,
    lossHistory: [],
    sidebarState: 'full', // 'full', 'mini', 'hidden' (mobile)

    init: function () {
        console.log("Initializing Run-Centric Dashboard...");
        this.initChart();
        this.updateHardwareInfo();
        this.restoreSidebarState(); // Restore from localStorage
        this.startPolling();
        this.viewHome();
    },

    startPolling: function () {
        if (this.pollInterval) clearInterval(this.pollInterval);
        this.pollInterval = setInterval(() => {
            this.checkActiveRun(); // Global check

            if (this.activeView === 'home') {
                this.updateRunsTable();
            } else if (this.activeView === 'run-detail' && this.activeRun) {
                this.updateRunMonitor();
            }
        }, 1000);
    },

    checkActiveRun: function () {
        fetch('/api/stats').then(r => r.json()).then(d => {
            const el = document.getElementById('active-run-display');
            if (d.run_name && d.run_name.length > 0) {
                el.style.display = 'block';
                document.getElementById('active-run-name').textContent = d.run_name;
            } else {
                el.style.display = 'none';
            }
        }).catch(() => { });
    },

    // --- NAVIGATION ---
    viewHome: function () {
        this.switchView('home');
        this.activeRun = null;
        this.updateRunsTable();
    },

    viewLibrary: function () {
        this.switchView('library');
        this.activeRun = null;
        this.fetchConfigs();
        this.fetchDatasets();
    },

    viewRun: function (name) {
        this.activeRun = name;
        this.switchView('run-detail');

        // Setup Header
        document.getElementById('detail-run-name').textContent = name;
        document.getElementById('detail-run-status').textContent = "Loading...";

        // Reset subtabs
        this.switchDetailTab('monitor');

        // Initial Fetch
        this.fetchRunDetails(name);
    },

    switchView: function (viewName) {
        this.activeView = viewName;
        // Sidebar active state
        document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
        if (viewName === 'home') document.querySelectorAll('.nav-item')[0].classList.add('active');
        if (viewName === 'library') document.querySelectorAll('.nav-item')[1].classList.add('active');

        // Show Section
        document.querySelectorAll('.view-section').forEach(el => el.classList.remove('active'));
        document.getElementById(`view-${viewName}`).classList.add('active');
    },

    switchDetailTab: function (tabName) {
        this.activeDetailTab = tabName;
        document.querySelectorAll('.detail-tab').forEach(el => el.classList.remove('active'));
        document.querySelectorAll('.detail-tab').forEach(el => {
            if (el.textContent.toLowerCase().includes(tabName) || el.onclick.toString().includes(tabName))
                el.classList.add('active');
        });

        document.querySelectorAll('.detail-content').forEach(el => el.style.display = 'none');
        document.getElementById(`detail-tab-${tabName}`).style.display = 'block';

        // Lazy Load
        if (tabName === 'config') this.fetchRunConfig();
        if (tabName === 'logs') this.fetchRunLogs();
        if (tabName === 'models') this.fetchRunModels();
    },

    // --- HOME WIDGETS ---
    updateRunsTable: function () {
        fetch('/api/runs').then(r => r.json()).then(runs => {
            const tbody = document.getElementById('runs-body');
            tbody.innerHTML = '';
            runs.forEach(run => {
                const tr = document.createElement('tr');
                tr.style.cursor = 'pointer';
                tr.onclick = () => this.viewRun(run.name);

                let statusColor = '#888';
                if (run.status === 'RUNNING') statusColor = '#2ecc71';
                else if (run.status === 'FAILED') statusColor = '#e74c3c';
                else if (run.status === 'COMPLETED') statusColor = '#3498db';

                tr.innerHTML = `
                    <td style="font-weight:bold; color:#fff;">${run.name}</td>
                    <td><span style="background:${statusColor}; color:#000; padding:2px 6px; border-radius:3px; font-size:0.8em; font-weight:bold;">${run.status}</span></td>
                    <td>${run.epoch}</td>
                    <td>${run.loss.toFixed(4)}</td>
                    <td>${run.size_mb} MB</td>
                `;
                tbody.appendChild(tr);
            });

            // Check Server Status
            document.getElementById('server-status').textContent = "Connected • " + runs.length + " runs";
            document.getElementById('server-status').style.color = "#2ecc71";
        }).catch(e => {
            document.getElementById('server-status').textContent = "Disconnected";
            document.getElementById('server-status').style.color = "#e74c3c";
        });
    },

    // --- RUN DETAILS ---
    fetchRunDetails: function (name) {
        // Get status to render buttons
        fetch('/api/runs').then(r => r.json()).then(runs => {
            const run = runs.find(r => r.name === name);
            if (run) {
                const statusEl = document.getElementById('detail-run-status');
                statusEl.textContent = run.status;
                statusEl.className = `run-status status-${run.status}`;

                const actions = document.getElementById('detail-actions');
                actions.innerHTML = '';

                if (run.status === 'RUNNING') {
                    actions.innerHTML = `<button class="btn-secondary" style="background:#e74c3c; color:white;" onclick="app.stopJob()">Stop Training</button>`;
                } else {
                    actions.innerHTML = `
                        <button class="btn-secondary" onclick="app.resumeJob('${name}')">Resume</button>
                        <button class="btn-secondary" style="background:#e74c3c; color:white;" onclick="app.deleteJob('${name}')">Delete</button>
                     `;
                }

                // Update Monitor Static Data if not running
                if (run.status !== 'RUNNING') {
                    document.getElementById('d-loss').textContent = run.loss.toFixed(4);
                    document.getElementById('d-step').textContent = "Stopped";
                    document.getElementById('d-speed').textContent = "0";
                    document.getElementById('d-lr').textContent = "-";
                }
            }
        });

        // Trigger tab load
        this.switchDetailTab(this.activeDetailTab);
    },

    updateRunMonitor: function () {
        // Only update live stats if we are watching the RUNNING job
        fetch('/api/runs').then(r => r.json()).then(runs => {
            const run = runs.find(r => r.name === this.activeRun);
            if (run && run.status === 'RUNNING') {
                fetch('/api/stats').then(r => r.json()).then(data => {
                    document.getElementById('d-loss').textContent = data.loss.toFixed(5);
                    document.getElementById('d-step').textContent = data.step;
                    document.getElementById('d-speed').textContent = data.speed.toFixed(1);
                    document.getElementById('d-lr').textContent = data.lr.toExponential(2);

                    if (data.history && data.history.length > 0) {
                        this.chart.data.labels = data.history.map((_, i) => i);
                        this.chart.data.datasets[0].data = data.history;
                        this.chart.update();
                    }
                });
            }
        });
    },

    fetchRunConfig: function () {
        if (!this.activeRun) return;
        const container = document.getElementById('d-config-content');
        container.textContent = "Loading...";

        fetch(`/api/runs/config?name=${this.activeRun}`)
            .then(r => r.json())
            .then(data => {
                if (data.error) container.textContent = data.error;
                else container.textContent = data.content || "Empty Config";
            });
    },

    fetchRunLogs: function () {
        if (!this.activeRun) return;
        const container = document.getElementById('d-log-content');
        // container.textContent = "Loading..."; // Don't wipe if refreshing

        fetch(`/api/runs/logs?name=${this.activeRun}`)
            .then(r => r.json())
            .then(data => {
                if (data.error) container.textContent = data.error;
                else {
                    container.textContent = data.content;
                    container.scrollTop = container.scrollHeight; // Auto scroll
                }
            });
    },

    fetchRunModels: function () {
        if (!this.activeRun) return;
        fetch('/api/models').then(r => r.json()).then(allModels => {
            // Filter for this run
            // Run models are in "run_name/checkpoint.bin"
            // Or maybe just check if name contains activeRun
            const models = allModels.filter(m => m.name.startsWith(this.activeRun + "/"));

            const list = document.getElementById('d-model-list');
            const select = document.getElementById('d-inference-model');
            list.innerHTML = '';
            select.innerHTML = '';

            if (models.length === 0) list.innerHTML = '<li style="color:#666">No checkpoints found.</li>';

            models.forEach(m => {
                // List Item
                const li = document.createElement('li');
                const shortName = m.name.split('/').pop();
                li.innerHTML = `<span>${shortName}</span> <a href="/api/models/download?name=${m.name}" target="_blank" class="btn-small btn-secondary">Download (${m.size_mb} MB)</a>`;
                list.appendChild(li);

                // Select Option
                const opt = document.createElement('option');
                opt.value = m.name;
                opt.textContent = shortName;
                select.appendChild(opt);
            });
        });
    },

    runRunInference: function () {
        const model = document.getElementById('d-inference-model').value;
        const prompt = document.getElementById('d-inference-prompt').value;
        const resultBox = document.getElementById('d-inference-result');
        if (!model) { alert("No model selected"); return; }

        resultBox.textContent = "Generating...";
        fetch('/api/inference', {
            method: 'POST',
            body: JSON.stringify({ model, prompt }),
            headers: { 'Content-Type': 'application/json' }
        }).then(r => r.json()).then(d => {
            resultBox.textContent = d.text || d.error;
        });
    },

    // --- ACTIONS ---
    stopJob: function () {
        if (!confirm("Stop training?")) return;
        fetch('/api/runs/stop', { method: 'POST' }).then(() => {
            setTimeout(() => this.fetchRunDetails(this.activeRun), 1000);
        });
    },
    resumeJob: function (name) {
        fetch('/api/runs/resume?name=' + name, { method: 'POST' }).then(() => {
            setTimeout(() => this.fetchRunDetails(this.activeRun), 1000);
        });
    },
    deleteJob: function (name) {
        if (!confirm("Delete run data permanently?")) return;
        fetch('/api/runs/delete?name=' + name, { method: 'DELETE' }).then(() => {
            this.viewHome(); // Go back
        });
    },

    // --- LIBRARY MANAGERS ---
    viewLibrary: function () {
        this.switchView('library');
        this.activeRun = null;
        this.fetchConfigsForEditor();
        this.fetchDatasets();
    },

    fetchConfigsForEditor: function () {
        fetch('/api/configs').then(r => r.json()).then(list => {
            const ul = document.getElementById('config-list-editor');
            if (ul) {
                ul.innerHTML = '';
                list.forEach(c => {
                    const li = document.createElement('li');
                    li.innerHTML = `<span>${c}</span>`;
                    li.style.cursor = 'pointer';
                    li.onclick = () => this.loadConfigToEditor(c);
                    ul.appendChild(li);
                });
            }
        });
    },

    loadConfigToEditor: function (name) {
        document.getElementById('editor-filename').textContent = name;
        document.getElementById('btn-save-as').disabled = false;

        const editor = document.getElementById('config-editor-content');
        editor.value = "Loading...";
        editor.disabled = true;

        fetch(`/api/configs/read?name=${name}`)
            .then(r => r.json())
            .then(data => {
                if (data.error) editor.value = "Error: " + data.error;
                else {
                    editor.value = data.content;
                    editor.disabled = false;
                }
            });
    },

    openSaveConfigModal: function () {
        document.getElementById('modal-save-config').style.display = 'block';
    },

    saveConfigAs: function () {
        const filename = document.getElementById('save-config-filename').value;
        const content = document.getElementById('config-editor-content').value;

        if (!filename) { alert("Enter filename"); return; }

        fetch('/api/configs/create', {
            method: 'POST',
            body: JSON.stringify({ filename: filename, content: content })
        }).then(r => r.json()).then(d => {
            if (d.error) alert(d.error);
            else {
                this.closeModal('modal-save-config');
                this.fetchConfigsForEditor(); // Refresh list
                this.loadConfigToEditor(d.file); // Select new file
                alert("Saved!");
            }
        });
    },
    fetchDatasets: function () {
        fetch('/api/datasets').then(r => r.json()).then(list => {
            const ul = document.getElementById('dataset-list');
            ul.innerHTML = '';
            list.forEach(c => {
                const li = document.createElement('li');
                li.innerHTML = `<span>${c.name}</span> <span style="color:#666">${c.size_mb} MB</span>`;
                ul.appendChild(li);
            });
        });
    },

    // --- CREATE/UPLOAD (New Run Modal) ---
    openNewRunModal: function () {
        document.getElementById('modal-new-run').style.display = 'block';
        // Populate dropdowns
        fetch('/api/configs').then(r => r.json()).then(l => {
            const s = document.getElementById('new-run-config'); s.innerHTML = '';
            l.forEach(x => { const o = document.createElement('option'); o.value = x; o.textContent = x; s.appendChild(o); });
        });

        // Ensure Customize button exists or add it dynamically if missing in HTML (Doing it in HTML is better)

        fetch('/api/datasets').then(r => r.json()).then(l => {
            const s = document.getElementById('new-run-dataset'); s.innerHTML = '';
            l.forEach(x => { const o = document.createElement('option'); o.value = x.name; o.textContent = x.name; s.appendChild(o); });
        });
    },
    startNewRun: function () {
        const name = document.getElementById('new-run-name').value;
        const config = document.getElementById('new-run-config').value;
        const dataset = document.getElementById('new-run-dataset').value;

        // Check for custom content
        const customEl = document.getElementById('new-run-custom-config');
        let content = null;
        if (customEl.style.display !== 'none' && customEl.value.trim().length > 0) {
            content = customEl.value;
        }

        if (!name) return alert("Enter name");

        fetch('/api/runs/start', {
            method: 'POST',
            body: JSON.stringify({
                run_name: name,
                config_file: config,
                data_file: dataset,
                config_content: content // Send content if edited
            })
        }).then(r => r.json()).then(d => {
            if (d.error) alert(d.error);
            else {
                document.getElementById('modal-new-run').style.display = 'none';
                this.viewRun(name); // Go close to the new run
            }
        });
    },

    toggleCustomConfig: function () {
        const el = document.getElementById('new-run-custom-config');
        const btn = document.getElementById('btn-toggle-custom');
        const select = document.getElementById('new-run-config');

        if (el.style.display === 'none') {
            el.style.display = 'block';
            btn.textContent = "Hide Config";
            // Load content if empty
            if (el.value.length === 0 && select.value) {
                this.loadTemplateContent(select.value);
            }
        } else {
            el.style.display = 'none';
            btn.textContent = "Show/Edit Config";
        }
    },

    onConfigSelectChange: function () {
        const el = document.getElementById('new-run-custom-config');
        const select = document.getElementById('new-run-config');
        if (el.style.display !== 'none') {
            this.loadTemplateContent(select.value);
        }
    },

    loadTemplateContent: function (name) {
        const el = document.getElementById('new-run-custom-config');
        el.value = "Loading...";
        el.disabled = true;
        fetch(`/api/configs/read?name=${name}`).then(r => r.json()).then(d => {
            el.value = d.content || "";
            el.disabled = false;
        });
    },

    customizeSelectedConfig: function () {
        // Deprecated/Removed in favor of inline toggle
    },

    // --- CHART ---
    initChart: function () {
        const ctx = document.getElementById('detail-chart').getContext('2d');
        this.chart = new Chart(ctx, {
            type: 'line',
            data: { labels: [], datasets: [{ label: 'Loss', data: [], borderColor: '#3498db', borderWidth: 2 }] },
            options: { responsive: true, maintainAspectRatio: false, scales: { x: { display: false }, y: { grid: { color: '#333' } } } }
        });
    },

    // --- MODALS (Generic) ---
    openNewConfigModal: function () { document.getElementById('modal-new-config').style.display = 'block'; },
    closeModal: function (id) { document.getElementById(id).style.display = 'none'; },
    createConfig: function () {
        const f = document.getElementById('new-config-filename').value;
        const c = document.getElementById('new-config-content').value;
        fetch('/api/configs/create', { method: 'POST', body: JSON.stringify({ filename: f, content: c }) })
            .then(() => { this.closeModal('modal-new-config'); this.fetchConfigs(); });
    },
    uploadDataset: function () {
        const f = document.getElementById('dataset-file-input').files[0];
        if (!f) return;
        document.getElementById('upload-status').textContent = 'Uploading...';
        fetch('/api/datasets/upload?name=' + f.name, { method: 'PUT', body: f }).then(() => {
            document.getElementById('upload-status').textContent = 'Done';
            this.fetchDatasets();
        });
    },

    updateHardwareInfo: function () {
        fetch('/api/hardware')
            .then(r => r.json())
            .then(hw => {
                const gpuEl = document.getElementById('hw-gpu');
                const memEl = document.getElementById('hw-mem');

                if (gpuEl) {
                    gpuEl.innerHTML = `<i class="fas fa-microchip" style="margin-right: 4px; color: #5e72e4;"></i> ${hw.compute_device || 'N/A'}`;
                }

                if (memEl) {
                    const vram_gb = (hw.mem_total_mb / 1024).toFixed(1);
                    memEl.textContent = `VRAM: ${vram_gb} GB`;
                }
            })
            .catch(err => {
                console.error('Failed to fetch hardware info:', err);
                const gpuEl = document.getElementById('hw-gpu');
                if (gpuEl) gpuEl.textContent = 'Hardware info unavailable';
            });
    },

    // Sidebar Control
    toggleSidebar: function () {
        const sidebar = document.getElementById('sidebar');
        const isMobile = window.innerWidth <= 768;

        if (isMobile) {
            // Mobile: toggle visibility
            sidebar.classList.toggle('mobile-open');
            this.sidebarState = sidebar.classList.contains('mobile-open') ? 'full' : 'hidden';
        } else {
            // Desktop: toggle between full and mini
            sidebar.classList.toggle('mini');
            this.sidebarState = sidebar.classList.contains('mini') ? 'mini' : 'full';
        }

        // Save to localStorage
        localStorage.setItem('sidebarState', this.sidebarState);
    },

    restoreSidebarState: function () {
        const saved = localStorage.getItem('sidebarState');
        const sidebar = document.getElementById('sidebar');
        const isMobile = window.innerWidth <= 768;

        if (saved && sidebar) {
            this.sidebarState = saved;

            if (isMobile) {
                // Mobile: default hidden
                if (saved === 'full') {
                    sidebar.classList.add('mobile-open');
                }
            } else {
                // Desktop: restore mini state
                if (saved === 'mini') {
                    sidebar.classList.add('mini');
                }
            }
        }
    }
};

document.addEventListener('DOMContentLoaded', () => app.init());
