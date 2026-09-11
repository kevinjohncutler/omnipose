"""Tests for omnipose.models.run_metadata — capture + write/update."""

import json
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from omnipose.models import run_metadata


# ---------------------------------------------------------------------------
# Pure capture functions
# ---------------------------------------------------------------------------

class TestCaptureProvenance:
    def test_returns_dict(self):
        prov = run_metadata.capture_provenance()
        assert isinstance(prov, dict)
        # These keys are always present (values may be None outside a git repo)
        for key in ('git_sha', 'git_branch', 'git_dirty', 'argv', 'cwd'):
            assert key in prov
        assert isinstance(prov['git_dirty'], bool)
        assert isinstance(prov['argv'], list)


class TestCaptureHardware:
    def test_cpu_device(self):
        hw = run_metadata.capture_hardware(torch.device('cpu'))
        assert hw['device'] == 'cpu'
        assert hw['device_type'] == 'cpu'
        assert 'torch_version' in hw
        assert 'python_version' in hw

    def test_string_device(self):
        hw = run_metadata.capture_hardware('cpu')
        assert hw['device'] == 'cpu'


class TestCaptureModelInfo:
    def test_captures_param_count(self):
        net = torch.nn.Linear(10, 3)
        info = run_metadata.capture_model_info(
            model=type('M', (), {})(),  # dummy
            net=net,
        )
        assert info['architecture'] == 'Linear'
        assert info['n_parameters'] == 10 * 3 + 3
        assert info['data_parallel'] is False


class TestCaptureDatasetInfo:
    def test_arrays(self):
        imgs = [np.zeros((4, 4)) for _ in range(3)]
        lbls = [np.zeros((4, 4), dtype=np.int32) for _ in range(3)]
        info = run_metadata.capture_dataset_info(imgs, lbls, None, None)
        assert info['n_train'] == 3
        assert info['n_test'] == 0
        assert info['train_mode'] == 'arrays'
        assert len(info['sample_image_shapes']) == 3

    def test_paths(self):
        info = run_metadata.capture_dataset_info(['a.tif', 'b.tif'], ['a_m.tif', 'b_m.tif'], None, None)
        assert info['train_mode'] == 'paths'
        assert 'sample_image_shapes' not in info  # not captured for paths

    def test_empty(self):
        info = run_metadata.capture_dataset_info(None, None, None, None)
        assert info['n_train'] == 0


class TestCaptureHyperparameters:
    def test_filters_non_primitives(self):
        model = type('M', (), {'omni': True, 'nchan': 1, 'nclasses': 4, 'dim': 2})()
        fake_locals = {
            'self': None,
            'learning_rate': 0.1,
            'n_epochs': 30,
            'tyx': (64, 64),
            'ignored_tensor': torch.zeros(3),  # should be filtered out
            '_private': 'skip',
        }
        hp = run_metadata.capture_hyperparameters(model, fake_locals)
        assert hp['learning_rate'] == 0.1
        assert hp['n_epochs'] == 30
        assert hp['tyx'] == [64, 64]  # tuple -> list
        assert '_private' not in hp
        assert 'ignored_tensor' not in hp
        assert hp['model.omni'] is True
        assert hp['model.nchan'] == 1
        assert isinstance(hp['loss_modules'], list)
        assert 'torch_seed' in hp


# ---------------------------------------------------------------------------
# compute_summary
# ---------------------------------------------------------------------------

class TestComputeSummary:
    def test_basic(self):
        history = {
            'epoch': [0, 0, 1, 1, 2, 2],
            'train_loss': [1.0, 0.8, 0.6, 0.5, 0.4, 0.3],
            'raw_losses': [{'a': 0.5, 'b': 0.5}] * 6,
        }
        s = run_metadata.compute_summary(history)
        assert s['final_loss_raw'] == pytest.approx(0.3)
        assert s['min_loss_raw'] == pytest.approx(0.3)
        assert s['min_loss_raw_epoch'] == 2
        assert s['total_batches'] == 6
        assert s['total_epochs_completed'] == 3
        assert s['n_loss_components'] == 2

    def test_empty(self):
        s = run_metadata.compute_summary({})
        assert s['final_loss_raw'] is None
        assert s['total_batches'] == 0


# ---------------------------------------------------------------------------
# Write / update
# ---------------------------------------------------------------------------

class TestWriteAndUpdate:
    def test_run_json_path_for(self, tmp_path):
        lh = tmp_path / 'run_abc_loss_history.json'
        rj = run_metadata.run_json_path_for(lh)
        assert rj.name == 'run_abc_run.json'
        assert rj.parent == tmp_path

    def test_write_round_trip(self, tmp_path):
        p = tmp_path / 'x_run.json'
        payload = {'schema_version': 1, 'name': 'test', 'hyperparameters': {'lr': 0.1}}
        run_metadata.write_run_json(p, payload)
        assert p.exists()
        loaded = json.loads(p.read_text())
        assert loaded == payload

    def test_atomic_write(self, tmp_path):
        """Write should not leave a .tmp file on success."""
        p = tmp_path / 'y_run.json'
        run_metadata.write_run_json(p, {'x': 1})
        tmp = p.with_suffix('.json.tmp')
        assert not tmp.exists()

    def test_update_merges(self, tmp_path):
        p = tmp_path / 'z_run.json'
        run_metadata.write_run_json(p, {'name': 'r', 'status': 'running'})
        run_metadata.update_run_json(p, {'status': 'completed', 'notes': 'ok'})
        loaded = json.loads(p.read_text())
        assert loaded['name'] == 'r'
        assert loaded['status'] == 'completed'
        assert loaded['notes'] == 'ok'

    def test_update_on_missing_file(self, tmp_path):
        """update should create from scratch if file missing."""
        p = tmp_path / 'new_run.json'
        run_metadata.update_run_json(p, {'foo': 'bar'})
        assert json.loads(p.read_text()) == {'foo': 'bar'}


class TestMarkRunFinished:
    def test_stamps_summary_and_status(self, tmp_path):
        p = tmp_path / 'a_run.json'
        run_metadata.write_run_json(p, {
            'schema_version': 1, 'name': 'a',
            'started_at': time.time() - 10, 'status': 'running',
        })
        history = {
            'epoch': [0, 0, 1, 1],
            'train_loss': [1.0, 0.8, 0.6, 0.5],
            'raw_losses': [{'a': 0.5}] * 4,
        }
        run_metadata.mark_run_finished(p, history)
        data = json.loads(p.read_text())
        assert data['status'] == 'completed'
        assert data['finished_at'] is not None
        assert data['duration_s'] > 0
        assert data['summary']['final_loss_raw'] == pytest.approx(0.5)

    def test_failure_status(self, tmp_path):
        p = tmp_path / 'b_run.json'
        run_metadata.write_run_json(p, {'started_at': time.time()})
        run_metadata.mark_run_finished(p, {}, status='failed')
        assert json.loads(p.read_text())['status'] == 'failed'


# ---------------------------------------------------------------------------
# Top-level capture_run_metadata
# ---------------------------------------------------------------------------

class TestCaptureRunMetadata:
    def test_end_to_end(self):
        model = type('M', (), {
            'omni': True, 'nchan': 1, 'nclasses': 4, 'dim': 2,
            'device': torch.device('cpu'),
        })()
        net = torch.nn.Linear(2, 1)

        imgs = [np.zeros((4, 4))]
        masks = [np.zeros((4, 4), dtype=np.int32)]

        fake_locals = {
            'learning_rate': 0.05, 'n_epochs': 10,
            'batch_size': 1, 'tyx': (32, 32),
        }

        payload = run_metadata.capture_run_metadata(
            model, net, fake_locals,
            imgs, masks, None, None,
            save_path='/tmp/x', netstr='foo',
            name='mytest', sweep='sweepA', tags=['demo'], notes='n/a',
        )

        assert payload['schema_version'] == 1
        assert payload['name'] == 'mytest'
        assert payload['sweep'] == 'sweepA'
        assert payload['tags'] == ['demo']
        assert payload['status'] == 'running'
        assert payload['hyperparameters']['learning_rate'] == 0.05
        assert payload['model']['architecture'] == 'Linear'
        assert payload['dataset']['n_train'] == 1
        assert 'hardware' in payload
        assert 'provenance' in payload
        assert payload['summary'] == {}
