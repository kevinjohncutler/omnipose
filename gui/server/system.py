"""System information utilities for the Omnipose GUI."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .segmentation import Segmenter


def get_system_info(segmenter: "Segmenter") -> dict[str, object]:
    """Get system information including RAM and GPU status."""
    total = None
    available = None
    used = None
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        total = int(vm.total)
        available = int(vm.available)
        used = int(vm.total - vm.available)
    except Exception:
        try:
            with open('/proc/meminfo', 'r', encoding='utf-8') as handle:
                data = handle.read().splitlines()
            meminfo = {}
            for line in data:
                parts = line.split(':', 1)
                if len(parts) != 2:
                    continue
                key = parts[0].strip()
                value = parts[1].strip().split()[0]
                meminfo[key] = int(value) * 1024
            total = meminfo.get('MemTotal')
            available = meminfo.get('MemAvailable')
            if total is not None and available is not None:
                used = int(total - available)
        except Exception:
            pass
    cpu_cores = os.cpu_count() or 1
    gpu_available = False
    gpu_name = None
    gpu_backend = None
    gpu_label = None
    try:
        from omnipose import gpu as omni_gpu  # type: ignore

        device, gpu_ok = omni_gpu.use_gpu(0, use_torch=True)
        gpu_available = bool(gpu_ok)
        gpu_backend = getattr(device, 'type', None)
        gpu_name = gpu_backend
        if gpu_available:
            if gpu_backend == 'cuda':
                try:
                    import torch  # type: ignore

                    gpu_name = torch.cuda.get_device_name(0)
                except Exception:
                    gpu_name = 'CUDA GPU'
            elif gpu_backend == 'mps':
                gpu_name = 'Apple MPS'
            else:
                gpu_name = 'GPU available'
        if gpu_available:
            # Only add backend prefix if gpu_name doesn't already include it
            if gpu_backend and gpu_name and gpu_backend.upper() not in gpu_name.upper():
                gpu_label = f"{gpu_backend.upper()}: {gpu_name}"
            elif gpu_name:
                gpu_label = str(gpu_name)
    except Exception:
        gpu_available = False
        gpu_name = None
        gpu_backend = None
        gpu_label = None
    return {
        'ram_total': total,
        'ram_available': available,
        'ram_used': used,
        'cpu_cores': cpu_cores,
        'gpu_available': gpu_available,
        'gpu_name': gpu_name,
        'gpu_backend': gpu_backend,
        'gpu_label': gpu_label,
        'use_gpu': segmenter.get_use_gpu(),
    }
