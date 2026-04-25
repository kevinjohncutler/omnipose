"""Captures and persists ``run.json`` — metadata sidecar for each training run.

Lives alongside ``*_loss_history.json`` so the viewer / sweep tools can filter
and compare runs by hyperparameter, model, dataset, or provenance without
parsing filenames or rescanning time-series data.

Schema version 1 — see docstring of :func:`capture_run_metadata`.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from hashlib import blake2b
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def _git(args, cwd):
    try:
        out = subprocess.check_output(
            ['git'] + list(args),
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return out.decode().strip()
    except Exception:
        return None


def _package_version(pkg_name):
    try:
        import importlib.metadata as md
        return md.version(pkg_name)
    except Exception:
        return None


def capture_provenance(source_file=None):
    """Git SHA, branch, dirty state, remote, package versions, argv, cwd."""
    # Use source_file's repo when provided; otherwise fall back to cwd
    repo_dir = Path(source_file).parent if source_file else Path.cwd()

    sha = _git(['rev-parse', 'HEAD'], repo_dir)
    branch = _git(['rev-parse', '--abbrev-ref', 'HEAD'], repo_dir)
    tag = _git(['describe', '--tags', '--always', '--dirty'], repo_dir)
    dirty_raw = _git(['status', '--porcelain'], repo_dir)
    remote = _git(['config', '--get', 'remote.origin.url'], repo_dir)

    return {
        'git_sha': sha,
        'git_branch': branch,
        'git_describe': tag,
        'git_dirty': bool(dirty_raw),
        'git_remote': remote,
        'omnipose_version': _package_version('omnipose'),
        'ocdkit_version': _package_version('ocdkit'),
        'argv': list(sys.argv),
        'cwd': str(Path.cwd()),
    }


# ---------------------------------------------------------------------------
# Hardware / platform
# ---------------------------------------------------------------------------

def capture_hardware(device):
    """torch/python/cuda/device info."""
    dev = device if isinstance(device, torch.device) else torch.device(device)
    info = {
        'device': str(dev),
        'device_type': dev.type,
        'torch_version': torch.__version__,
        'python_version': sys.version.split()[0],
        'platform': platform.platform(),
        'cuda_available': torch.cuda.is_available(),
        'mps_available': (hasattr(torch.backends, 'mps')
                          and torch.backends.mps.is_available()),
        'n_gpu': 0,
        'gpu_name': None,
        'cuda_version': None,
    }
    if torch.cuda.is_available():
        info['n_gpu'] = torch.cuda.device_count()
        try:
            info['gpu_name'] = torch.cuda.get_device_name(0)
        except Exception:
            pass
        info['cuda_version'] = getattr(torch.version, 'cuda', None)
    return info


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def capture_model_info(model, net):
    """Architecture summary for the model instance + network."""
    inner = net.module if isinstance(net, torch.nn.DataParallel) else net
    n_params = sum(p.numel() for p in inner.parameters())

    return {
        'architecture': type(inner).__name__,
        'nbase': list(getattr(inner, 'nbase', []) or []),
        'sz': getattr(inner, 'sz', None),
        'residual_on': getattr(inner, 'residual_on', None),
        'style_on': getattr(inner, 'style_on', None),
        'concatenation': getattr(inner, 'concatenation', None),
        'dim': getattr(inner, 'dim', None),
        'kernel_size': getattr(inner, 'kernel_size', None),
        'scale_factor': getattr(inner, 'scale_factor', None),
        'dilation': getattr(inner, 'dilation', None),
        'n_parameters': int(n_params),
        'data_parallel': isinstance(net, torch.nn.DataParallel),
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def capture_dataset_info(train_data, train_labels, test_data, test_labels,
                         save_path=None):
    """Counts + sample shapes. Data paths if available."""
    def _shape(x):
        return tuple(x.shape) if hasattr(x, 'shape') else None

    def _path_type(data):
        if not data:
            return None
        if isinstance(data[0], (str, os.PathLike)):
            return 'paths'
        return 'arrays'

    info = {
        'n_train': len(train_data) if train_data is not None else 0,
        'n_test': len(test_data) if test_data is not None else 0,
        'train_mode': _path_type(train_data),
        'test_mode': _path_type(test_data) if test_data else None,
        'save_path': str(save_path) if save_path else None,
    }

    # Sample up to 8 shapes so this stays bounded
    sample_n = min(8, info['n_train'])
    if sample_n and _path_type(train_data) == 'arrays':
        info['sample_image_shapes'] = [_shape(train_data[i]) for i in range(sample_n)]
        if train_labels is not None:
            info['sample_mask_shapes'] = [_shape(train_labels[i]) for i in range(sample_n)]
    return info


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

# Keys we pull from the model instance (effective values, not defaults)
_MODEL_ATTR_KEYS = (
    'omni', 'nchan', 'nclasses', 'dim', 'diam_mean',
    'torch_compile', 'norm_type',
)


def capture_hyperparameters(model, train_locals):
    """Merge ``_train_net`` locals with model instance attrs.

    Captures every resolved value — so if ``num_workers=-1`` on CLI but got
    auto-resolved to 2, we record 2 (not -1).
    """
    # Extract all scalar / sequence / primitive fields from locals
    keep = (int, float, bool, str, type(None), tuple, list)
    hp = {}
    for k, v in train_locals.items():
        if k.startswith('_') or k == 'self':
            continue
        if isinstance(v, keep):
            hp[k] = list(v) if isinstance(v, tuple) else v
        elif isinstance(v, torch.device):
            hp[k] = str(v)

    # Model attributes (effective config)
    for key in _MODEL_ATTR_KEYS:
        if hasattr(model, key):
            val = getattr(model, key)
            if isinstance(val, torch.device):
                val = str(val)
            hp[f'model.{key}'] = val

    # Names of active loss modules on the model, if any
    loss_names = []
    for attr in dir(model):
        if attr.lower().endswith('loss') and isinstance(getattr(model, attr, None),
                                                         torch.nn.Module):
            loss_names.append(attr)
    hp['loss_modules'] = sorted(loss_names)

    # Torch RNG seed in effect at capture time. Useful even if the user didn't
    # explicitly set one — captures the auto-chosen seed for reproducibility.
    try:
        hp['torch_seed'] = int(torch.initial_seed())
    except Exception:
        hp['torch_seed'] = None

    return hp


# ---------------------------------------------------------------------------
# Aggregation / summary
# ---------------------------------------------------------------------------

def compute_summary(loss_history):
    """Extract final/min loss stats from a completed loss history."""
    train_loss = loss_history.get('train_loss') or []
    epochs = loss_history.get('epoch') or []
    raw_losses = loss_history.get('raw_losses') or []

    if not train_loss:
        return {
            'final_loss_raw': None,
            'min_loss_raw': None,
            'min_loss_raw_epoch': None,
            'total_batches': 0,
            'total_epochs_completed': 0,
        }

    arr = np.array([v for v in train_loss if v is not None], dtype=np.float64)
    idx_min = int(np.argmin(arr)) if len(arr) else None

    return {
        'final_loss_raw': float(train_loss[-1]),
        'min_loss_raw': float(arr.min()) if len(arr) else None,
        'min_loss_raw_epoch': int(epochs[idx_min]) if idx_min is not None else None,
        'total_batches': len(train_loss),
        'total_epochs_completed': (max(epochs) + 1) if epochs else 0,
        'n_loss_components': len(raw_losses[0]) if raw_losses and raw_losses[0] else 0,
    }


# ---------------------------------------------------------------------------
# Public API: write / update run.json
# ---------------------------------------------------------------------------

def run_json_path_for(loss_history_path):
    """``foo_loss_history.json`` -> ``foo_run.json`` (same directory)."""
    p = Path(loss_history_path)
    stem = p.stem.removesuffix('_loss_history')
    return p.with_name(f'{stem}_run.json')


def _run_id(started_at, name, sweep):
    return blake2b(f'{started_at}:{name}:{sweep}'.encode(), digest_size=6).hexdigest()


def capture_run_metadata(model, net, train_locals,
                         train_data, train_labels,
                         test_data, test_labels,
                         save_path, netstr,
                         name=None, sweep=None, tags=None, notes=None):
    """Build the full ``run.json`` payload (schema version 1).

    ``name``/``sweep``/``tags``/``notes`` are optional user-supplied annotations.
    """
    started_at = time.time()
    effective_name = name or netstr or 'run'

    device = getattr(model, 'device', train_locals.get('device'))

    return {
        'schema_version': SCHEMA_VERSION,
        'run_id': _run_id(started_at, effective_name, sweep or ''),
        'name': effective_name,
        'sweep': sweep,
        'tags': list(tags) if tags else [],
        'status': 'running',
        'started_at': started_at,
        'finished_at': None,
        'duration_s': None,
        'hyperparameters': capture_hyperparameters(model, train_locals),
        'model': capture_model_info(model, net),
        'dataset': capture_dataset_info(train_data, train_labels,
                                        test_data, test_labels,
                                        save_path=save_path),
        'hardware': capture_hardware(device),
        'provenance': capture_provenance(),
        'summary': {},
        'notes': notes or '',
    }


def write_run_json(path, payload):
    """Atomic write (temp + rename) so the viewer never sees partial JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(payload, f, default=_json_default, indent=2)
    os.replace(tmp, path)


def update_run_json(path, updates):
    """Merge ``updates`` into the existing ``run.json``, atomic write."""
    path = Path(path)
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}
    data.update(updates)
    write_run_json(path, data)


def mark_run_finished(path, loss_history, status='completed'):
    """Called at end-of-training to stamp duration + summary."""
    now = time.time()
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        return
    data['status'] = status
    data['finished_at'] = now
    data['duration_s'] = now - data.get('started_at', now)
    data['summary'] = compute_summary(loss_history)
    write_run_json(path, data)


# ---------------------------------------------------------------------------

def _json_default(obj):
    """Tolerant JSON encoder for numpy scalars, Paths, devices."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    return str(obj)
