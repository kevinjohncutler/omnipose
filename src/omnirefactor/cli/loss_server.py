"""
Live training loss viewer with auto-refresh using uPlot.

Usage:
    python -m omnirefactor.cli.loss_server /path/to/model_loss_history.json
    python -m omnirefactor.cli.loss_server /path/to/save_dir --port 8080
"""

import argparse
import os
import sys
import threading
import time
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

HTML_CONTENT = r'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Loss Viewer</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/uplot@1.6.30/dist/uPlot.min.css">
    <script src="https://cdn.jsdelivr.net/npm/uplot@1.6.30/dist/uPlot.iife.min.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: system-ui, -apple-system, sans-serif; background: #0d1117; color: #c9d1d9; }
        .container { max-width: 1400px; margin: 0 auto; padding: 16px; }
        h1 { font-size: 20px; font-weight: 500; margin-bottom: 12px; color: #58a6ff; }
        .controls {
            display: flex; gap: 16px; align-items: center; flex-wrap: wrap;
            background: #161b22; padding: 12px 16px; border-radius: 6px; margin-bottom: 12px;
        }
        .control { display: flex; align-items: center; gap: 6px; }
        label { font-size: 13px; color: #8b949e; }
        select, input {
            background: #21262d; border: 1px solid #30363d; color: #c9d1d9;
            padding: 4px 8px; border-radius: 4px; font-size: 13px;
        }
        input[type="number"] { width: 50px; }
        .status { margin-left: auto; font-size: 12px; color: #8b949e; }
        .status::before { content: ''; display: inline-block; width: 8px; height: 8px;
            background: #3fb950; border-radius: 50%; margin-right: 6px; }
        #chart { background: #0d1117; border-radius: 6px; }
        .legend {
            display: flex; flex-wrap: wrap; gap: 8px 16px;
            padding: 12px 16px; background: #161b22; border-radius: 6px; margin-top: 12px;
        }
        .legend-item { display: flex; align-items: center; gap: 6px; font-size: 12px; }
        .legend-color { width: 12px; height: 3px; border-radius: 1px; }
        .legend-value { font-family: monospace; color: #8b949e; }
    </style>
</head>
<body>
<div class="container">
    <h1>Training Loss Viewer</h1>
    <div class="controls">
        <div class="control">
            <label>Normalize:</label>
            <select id="norm">
                <option value="minmax">Min-Max (0-1)</option>
                <option value="raw">Raw</option>
                <option value="log">Log</option>
                <option value="zscore">Z-Score</option>
            </select>
        </div>
        <div class="control">
            <label>Smooth:</label>
            <input type="number" id="smooth" value="1" min="1" max="50">
        </div>
        <div class="control">
            <label>X:</label>
            <select id="xaxis">
                <option value="step">Step</option>
                <option value="epoch">Epoch</option>
            </select>
        </div>
        <div class="status" id="status">Connecting...</div>
    </div>
    <div id="chart"></div>
    <div class="legend" id="legend"></div>
</div>
<script>
const COLORS = ['#58a6ff','#f778ba','#7ee787','#ffa657','#d2a8ff','#79c0ff','#ff7b72','#a5d6ff','#ffd33d','#8b949e'];
let raw = null, plot = null;

const norm = document.getElementById('norm');
const smooth = document.getElementById('smooth');
const xaxis = document.getElementById('xaxis');
const status = document.getElementById('status');

[norm, smooth, xaxis].forEach(el => el.addEventListener('change', render));

function smoothArr(arr, w) {
    if (w <= 1) return arr;
    const out = new Array(arr.length);
    for (let i = 0; i < arr.length; i++) {
        let sum = 0, cnt = 0;
        for (let j = Math.max(0, i - w + 1); j <= i; j++) {
            if (arr[j] != null) { sum += arr[j]; cnt++; }
        }
        out[i] = cnt > 0 ? sum / cnt : null;
    }
    return out;
}

function normalize(arr, mode) {
    const valid = arr.filter(v => v != null && !isNaN(v));
    if (!valid.length) return arr;

    if (mode === 'raw') return arr;

    if (mode === 'minmax') {
        const min = Math.min(...valid), max = Math.max(...valid), rng = max - min;
        return rng < 1e-12 ? arr.map(v => v == null ? null : 0.5) : arr.map(v => v == null ? null : (v - min) / rng);
    }
    if (mode === 'log') {
        const minPos = Math.min(...valid.filter(v => v > 0)) || 1e-10;
        return arr.map(v => v == null ? null : Math.log10(Math.max(v, minPos)));
    }
    if (mode === 'zscore') {
        const mean = valid.reduce((a,b) => a+b, 0) / valid.length;
        const std = Math.sqrt(valid.reduce((a,b) => a + (b-mean)**2, 0) / valid.length) || 1;
        return arr.map(v => v == null ? null : (v - mean) / std);
    }
    return arr;
}

function render() {
    if (!raw) return;

    const mode = norm.value;
    const sw = parseInt(smooth.value) || 1;
    const xm = xaxis.value;

    const losses = raw.raw_losses || [];
    const epochs = raw.epoch || [];
    const batches = raw.batch || [];
    const n = losses.length;

    // Steps per epoch
    const ec = {};
    epochs.forEach(e => ec[e] = (ec[e]||0) + 1);
    const spe = Math.max(...Object.values(ec), 1);

    // X values
    const xs = new Array(n);
    for (let i = 0; i < n; i++) {
        const e = epochs[i] || 0, b = batches[i] || (i % spe);
        xs[i] = xm === 'epoch' ? e + b/spe : e * spe + b;
    }

    // Collect series names
    const names = new Set();
    losses.forEach(entry => { if (entry) Object.keys(entry).forEach(k => names.add(k)); });
    const nameList = [...names];

    // Build data array: [xs, series1, series2, ...]
    const data = [xs];
    const series = [{}]; // x-axis config
    const legendEl = document.getElementById('legend');
    legendEl.innerHTML = '';

    nameList.forEach((name, i) => {
        const vals = losses.map(entry => entry?.[name] ?? null);
        const normed = normalize(vals, mode);
        const smoothed = smoothArr(normed, sw);
        data.push(smoothed);

        const color = COLORS[i % COLORS.length];
        series.push({ stroke: color, width: 1.5, label: name });

        // Legend
        const lastVal = vals.filter(v => v != null).pop();
        const div = document.createElement('div');
        div.className = 'legend-item';
        div.innerHTML = `<span class="legend-color" style="background:${color}"></span>
            <span>${name}</span>
            <span class="legend-value">${lastVal != null ? lastVal.toExponential(2) : '-'}</span>`;
        legendEl.appendChild(div);
    });

    // Add total loss
    if (raw.train_loss?.length) {
        const normed = normalize(raw.train_loss, mode);
        const smoothed = smoothArr(normed, sw);
        data.push(smoothed);
        series.push({ stroke: '#fff', width: 2, dash: [4,2], label: 'total' });
    }

    const opts = {
        width: document.getElementById('chart').clientWidth || 1200,
        height: 450,
        series: series,
        scales: { x: { time: false } },
        axes: [
            { stroke: '#8b949e', grid: { stroke: '#21262d' }, ticks: { stroke: '#30363d' } },
            { stroke: '#8b949e', grid: { stroke: '#21262d' }, ticks: { stroke: '#30363d' } }
        ],
        cursor: { sync: { key: 'loss' } },
        legend: { show: false }
    };

    if (plot) plot.destroy();
    document.getElementById('chart').innerHTML = '';
    plot = new uPlot(opts, data, document.getElementById('chart'));
}

async function fetchData() {
    try {
        const r = await fetch('/data?t=' + Date.now());
        if (!r.ok) throw new Error('HTTP ' + r.status);
        const d = await r.json();
        const changed = !raw || d.train_loss?.length !== raw.train_loss?.length;
        raw = d;
        if (changed) render();
        const ep = d.epoch?.[d.epoch.length-1] || 0;
        status.textContent = `Epoch ${ep} • ${d.train_loss?.length || 0} batches`;
    } catch (e) {
        status.textContent = 'Error: ' + e.message;
    }
}

window.addEventListener('resize', () => { if (plot) render(); });
fetchData();
setInterval(fetchData, 1000);
</script>
</body>
</html>
'''


class LossViewerHandler(SimpleHTTPRequestHandler):
    json_path = None

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ('/', '/index.html'):
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode())
        elif path == '/data':
            self.serve_json()
        else:
            self.send_error(404)

    def serve_json(self):
        if not self.json_path or not os.path.exists(self.json_path):
            self.send_error(404, 'Loss history not found')
            return
        try:
            with open(self.json_path, 'r') as f:
                data = f.read()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(data.encode())
        except Exception as e:
            self.send_error(500, str(e))


def find_loss_history(path):
    path = Path(path)
    if path.is_file() and path.suffix == '.json':
        return str(path)
    if path.is_dir():
        for pattern in ['*_loss_history.json', '*loss*.json']:
            files = list(path.glob(pattern))
            if files:
                return str(max(files, key=lambda p: p.stat().st_mtime))
        models = path / 'models'
        if models.exists():
            for pattern in ['*_loss_history.json', '*loss*.json']:
                files = list(models.glob(pattern))
                if files:
                    return str(max(files, key=lambda p: p.stat().st_mtime))
    return None


def main():
    parser = argparse.ArgumentParser(description='Live loss viewer with uPlot')
    parser.add_argument('path', help='Path to loss_history.json or directory')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()

    json_path = find_loss_history(args.path)
    if not json_path:
        print(f"No loss history JSON found at {args.path}", file=sys.stderr)
        sys.exit(1)

    LossViewerHandler.json_path = json_path
    server = HTTPServer((args.host, args.port), LossViewerHandler)
    url = f'http://{args.host}:{args.port}'

    print(f'Loss Viewer: {url}')
    print(f'Data: {json_path}')
    print('Ctrl+C to stop\n')

    if not args.no_browser:
        threading.Thread(target=lambda: (time.sleep(0.3), webbrowser.open(url)), daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopped.')


if __name__ == '__main__':
    main()
