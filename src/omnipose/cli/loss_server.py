"""Live training loss viewer with session browser.

Usage:
    omnipose loss-server /path/to/models            # scan a directory
    omnipose loss-server /path/to/loss.json         # single session
    python -m omnipose.cli.loss_server . --port 8080
"""

import argparse
import json
import os
import sys
import threading
import time
import webbrowser
from hashlib import blake2b
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

HTML_PATH = Path(__file__).with_name('loss_viewer.html')


def _session_id(path):
    """Stable short id derived from absolute path."""
    return blake2b(str(Path(path).resolve()).encode(), digest_size=6).hexdigest()


def _scan_sessions(root):
    """Return list of session dicts: id, name, path, mtime, size."""
    root = Path(root).expanduser().resolve()
    sessions = []

    if root.is_file():
        paths = [root] if root.suffix == '.json' else []
    elif root.is_dir():
        paths = list(root.rglob('*_loss_history.json'))
        if not paths:
            paths = list(root.rglob('*loss*.json'))
    else:
        paths = []

    for p in paths:
        try:
            stat = p.stat()
        except OSError:
            continue
        name = p.stem.removesuffix('_loss_history')
        sessions.append({
            'id': _session_id(p),
            'name': name,
            'path': str(p),
            'mtime': stat.st_mtime,
            'size': stat.st_size,
        })

    sessions.sort(key=lambda s: s['mtime'], reverse=True)
    return sessions


def _run_json_path(loss_history_path):
    """``foo_loss_history.json`` -> ``foo_run.json`` (same dir)."""
    p = Path(loss_history_path)
    stem = p.stem.removesuffix('_loss_history')
    return p.with_name(f'{stem}_run.json')


def _load_run_json(session):
    """Return the sidecar run.json payload if present, else None."""
    p = _run_json_path(session['path'])
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def _session_summary(session):
    """Load a session's JSON and attach summary fields (epoch count, final losses)."""
    try:
        with open(session['path']) as f:
            data = json.load(f)
    except Exception:
        return session

    epochs = data.get('epoch') or []
    train_loss = data.get('train_loss') or []
    raw_losses = data.get('raw_losses') or []
    run = _load_run_json(session)

    return {
        **session,
        'n_batches': len(train_loss),
        'n_epochs': (max(epochs) + 1) if epochs else 0,
        'last_loss': train_loss[-1] if train_loss else None,
        'loss_names': sorted({k for entry in raw_losses if entry for k in entry.keys()}),
        'run': run,  # may be None for pre-schema sessions
    }


class LossViewerHandler(SimpleHTTPRequestHandler):
    root_path = None

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        path = urlparse(self.path).path

        if path in ('/', '/index.html'):
            self._serve_html()
        elif path == '/api/sessions':
            self._serve_sessions()
        elif path.startswith('/api/session/'):
            self._serve_session(path.rsplit('/', 1)[-1])
        else:
            self.send_error(404)

    def _serve_html(self):
        try:
            html = HTML_PATH.read_bytes()
        except OSError as exc:
            self.send_error(500, f'UI not found: {exc}')
            return
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(html)

    def _serve_sessions(self):
        sessions = [_session_summary(s) for s in _scan_sessions(self.root_path)]
        self._send_json(sessions)

    def _serve_session(self, session_id):
        for s in _scan_sessions(self.root_path):
            if s['id'] == session_id:
                try:
                    with open(s['path']) as f:
                        payload = json.load(f)
                except Exception as exc:
                    self.send_error(500, f'Failed to load session: {exc}')
                    return
                payload['_session'] = _session_summary(s)
                self._send_json(payload)
                return
        self.send_error(404, 'Session not found')

    def _send_json(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description='Training loss viewer with session browser')
    parser.add_argument('path', nargs='?', default='.',
                        help='Directory to scan for *_loss_history.json, or a specific JSON file')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()

    root = Path(args.path).expanduser().resolve()
    if not root.exists():
        print(f'Path not found: {root}', file=sys.stderr)
        sys.exit(1)

    LossViewerHandler.root_path = str(root)
    sessions = _scan_sessions(root)

    server = HTTPServer((args.host, args.port), LossViewerHandler)
    url = f'http://{args.host}:{args.port}'

    print(f'Loss Viewer: {url}')
    print(f'Scanning: {root}')
    if sessions:
        print(f'Found {len(sessions)} session(s)')
    else:
        print('No sessions yet — waiting for training to write loss history files')
    print('Ctrl+C to stop\n')

    if not args.no_browser:
        threading.Thread(
            target=lambda: (time.sleep(0.3), webbrowser.open(url)),
            daemon=True,
        ).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopped.')


if __name__ == '__main__':
    main()
