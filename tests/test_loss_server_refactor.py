"""Tests for omnirefactor.cli.loss_server — session scanning + HTTP endpoints."""

import http.client
import json
import threading
import time
from pathlib import Path

import pytest

from omnirefactor.cli import loss_server


def _write_session(path, name, n_epochs=3, n_batches=5, loss_names=('a', 'b')):
    data = {
        'epoch': [], 'batch': [], 'train_loss': [], 'epoch_loss': [],
        'learning_rate': [], 'timestamp': [], 'raw_losses': [],
    }
    for e in range(n_epochs):
        for b in range(n_batches):
            data['epoch'].append(e)
            data['batch'].append(b)
            data['train_loss'].append(1.0 / (1 + e * 0.1))
            data['epoch_loss'].append(None if b < n_batches - 1 else 0.5)
            data['learning_rate'].append(0.2)
            data['timestamp'].append(0)
            data['raw_losses'].append({n: 0.3 * (i + 1) for i, n in enumerate(loss_names)})
    p = Path(path) / f'{name}_loss_history.json'
    p.write_text(json.dumps(data))
    return p


# ---------------------------------------------------------------------------
# _scan_sessions
# ---------------------------------------------------------------------------

class TestScanSessions:
    def test_empty_dir(self, tmp_path):
        assert loss_server._scan_sessions(tmp_path) == []

    def test_finds_single_json(self, tmp_path):
        _write_session(tmp_path, 'run_a')
        sessions = loss_server._scan_sessions(tmp_path)
        assert len(sessions) == 1
        assert sessions[0]['name'] == 'run_a'

    def test_single_file_path(self, tmp_path):
        p = _write_session(tmp_path, 'run_a')
        sessions = loss_server._scan_sessions(p)
        assert len(sessions) == 1

    def test_sorted_by_mtime_desc(self, tmp_path):
        p1 = _write_session(tmp_path, 'old')
        time.sleep(0.01)
        p2 = _write_session(tmp_path, 'new')
        sessions = loss_server._scan_sessions(tmp_path)
        assert sessions[0]['name'] == 'new'
        assert sessions[1]['name'] == 'old'

    def test_recurses_subdirs(self, tmp_path):
        sub = tmp_path / 'deep'
        sub.mkdir()
        _write_session(sub, 'nested')
        sessions = loss_server._scan_sessions(tmp_path)
        assert len(sessions) == 1
        assert sessions[0]['name'] == 'nested'

    def test_nonexistent_path_returns_empty(self, tmp_path):
        assert loss_server._scan_sessions(tmp_path / 'missing') == []

    def test_stable_id(self, tmp_path):
        p = _write_session(tmp_path, 'run_x')
        sid1 = loss_server._session_id(p)
        sid2 = loss_server._session_id(p)
        assert sid1 == sid2


# ---------------------------------------------------------------------------
# _session_summary
# ---------------------------------------------------------------------------

class TestSessionSummary:
    def test_populates_counts(self, tmp_path):
        _write_session(tmp_path, 'r', n_epochs=4, n_batches=7, loss_names=('x', 'y', 'z'))
        session = loss_server._scan_sessions(tmp_path)[0]
        s = loss_server._session_summary(session)
        assert s['n_epochs'] == 4
        assert s['n_batches'] == 4 * 7
        assert s['loss_names'] == ['x', 'y', 'z']
        assert s['last_loss'] is not None

    def test_missing_keys_fallback(self, tmp_path):
        (tmp_path / 'empty_loss_history.json').write_text('{}')
        session = loss_server._scan_sessions(tmp_path)[0]
        s = loss_server._session_summary(session)
        assert s['n_epochs'] == 0
        assert s['n_batches'] == 0
        assert s['last_loss'] is None


# ---------------------------------------------------------------------------
# HTTP endpoints (real server on a random port)
# ---------------------------------------------------------------------------

@pytest.fixture
def live_server(tmp_path):
    """Spin up the loss_server on a random port, teardown on exit."""
    _write_session(tmp_path, 'alpha')
    _write_session(tmp_path, 'beta', n_epochs=5, n_batches=3, loss_names=('flow', 'dist'))

    loss_server.LossViewerHandler.root_path = str(tmp_path)
    server = loss_server.HTTPServer(('localhost', 0), loss_server.LossViewerHandler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield port
    finally:
        server.shutdown()


def _request(port, path):
    conn = http.client.HTTPConnection('localhost', port, timeout=5)
    conn.request('GET', path)
    r = conn.getresponse()
    body = r.read()
    conn.close()
    return r.status, body


class TestHttpEndpoints:
    def test_index_serves_html(self, live_server):
        status, body = _request(live_server, '/')
        assert status == 200
        assert b'<!DOCTYPE html>' in body
        assert b'Loss Viewer' in body

    def test_sessions_api(self, live_server):
        status, body = _request(live_server, '/api/sessions')
        assert status == 200
        sessions = json.loads(body)
        assert len(sessions) == 2
        names = {s['name'] for s in sessions}
        assert names == {'alpha', 'beta'}
        # Each session has summary fields
        for s in sessions:
            assert 'n_epochs' in s and 'n_batches' in s and 'loss_names' in s

    def test_session_detail_by_id(self, live_server):
        _, sessions_body = _request(live_server, '/api/sessions')
        sessions = json.loads(sessions_body)
        target = next(s for s in sessions if s['name'] == 'beta')

        status, body = _request(live_server, f'/api/session/{target["id"]}')
        assert status == 200
        data = json.loads(body)
        assert data['_session']['name'] == 'beta'
        assert 'train_loss' in data
        assert 'raw_losses' in data
        assert data['_session']['loss_names'] == ['dist', 'flow']

    def test_unknown_session_returns_404(self, live_server):
        status, _ = _request(live_server, '/api/session/deadbeef')
        assert status == 404

    def test_unknown_path_returns_404(self, live_server):
        status, _ = _request(live_server, '/totally/unknown')
        assert status == 404


class TestHttpEmptyDir:
    def test_sessions_api_empty(self, tmp_path):
        """Server starts cleanly with no sessions (live-training use case)."""
        loss_server.LossViewerHandler.root_path = str(tmp_path)
        server = loss_server.HTTPServer(('localhost', 0), loss_server.LossViewerHandler)
        port = server.server_port
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            status, body = _request(port, '/api/sessions')
            assert status == 200
            assert json.loads(body) == []
        finally:
            server.shutdown()
