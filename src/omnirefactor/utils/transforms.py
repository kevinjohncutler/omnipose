from __future__ import annotations

# Compatibility shim: keep utils.transforms API but source implementations from transforms/.

from ..transforms.modules import get_module
from ..transforms.normalize import rescale, safe_divide
from ..transforms.vector import torch_norm
from ..transforms.augment import rotate
from ..transforms.normalize import (
    adjust_contrast_masked,
    auto_chunked_quantile,
    bin_counts,
    compute_density,
    compute_quantiles,
    gamma_normalize,
    localnormalize,
    localnormalize_GPU,
    normalize99,
    normalize99_hist,
    normalize_field,
    normalize_image,
    pnormalize,
    qnorm,
    quantile_rescale,
    searchsorted,
)

import torch

TORCH_ENABLED = True

__all__ = [
    "get_module",
    "rescale",
    "safe_divide",
    "torch_norm",
    "rotate",
    "adjust_contrast_masked",
    "auto_chunked_quantile",
    "bin_counts",
    "compute_density",
    "compute_quantiles",
    "gamma_normalize",
    "localnormalize",
    "localnormalize_GPU",
    "normalize99",
    "normalize99_hist",
    "normalize_field",
    "normalize_image",
    "pnormalize",
    "qnorm",
    "quantile_rescale",
    "searchsorted",
    "TORCH_ENABLED",
]
