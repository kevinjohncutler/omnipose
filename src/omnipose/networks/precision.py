"""Device assignment and CUDA precision lock for reproducible numerics."""

from .imports import *


_CUDA_PRECISION_LOCKED = False
_ALLOW_TF32_ENV = os.environ.get("CELLPOSE_OMNI_ALLOW_TF32", "").lower() in {"1", "true", "yes", "on"}


def _lock_cuda_precision(device):
    """Disable TF32/autotuned kernels so CUDA numerics stay aligned with CPU."""
    global _CUDA_PRECISION_LOCKED

    if _ALLOW_TF32_ENV:
        if not _CUDA_PRECISION_LOCKED:
            core_logger.info(
                "CELLPOSE_OMNI_ALLOW_TF32 set; leaving CUDA TF32/autotune enabled for benchmarking."
            )
            _CUDA_PRECISION_LOCKED = True
        return

    if _CUDA_PRECISION_LOCKED:
        return

    if not isinstance(device, torch.device) or device.type != 'cuda':
        return

    cudnn = getattr(torch.backends, 'cudnn', None)
    if cudnn is not None:
        cudnn.deterministic = True
        cudnn.benchmark = False
        if hasattr(cudnn, 'allow_tf32'):
            cudnn.allow_tf32 = False

    cuda_matmul = getattr(torch.backends, 'cuda', None)
    if cuda_matmul is not None:
        matmul = getattr(cuda_matmul, 'matmul', None)
        if matmul is not None and hasattr(matmul, 'allow_tf32'):
            matmul.allow_tf32 = False

    if hasattr(torch, 'set_float32_matmul_precision'):
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass

    _CUDA_PRECISION_LOCKED = True
    core_logger.info('Enforcing deterministic FP32 CUDA kernels for reproducibility.')


def assign_device(gpu=True, gpu_number=None):
    device, gpu_available = get_device(gpu_number)
    if gpu and gpu_available:
        core_logger.info('Using GPU.')
        _lock_cuda_precision(device)
    elif gpu and not gpu_available:
        core_logger.info('No GPU available or pytorch not configured, using CPU.')
        device = torch_CPU
    else:
        core_logger.info('Using CPU.')
        device = torch_CPU
    return device, gpu_available
