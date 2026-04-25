"""Centralized imports for the core subpackage (Layer 2).

Layer 2: depends on L0 (utils, gpu) and L1 (transforms) — never io or models.
"""

import time

import numpy as np
import torch
import scipy
from scipy.ndimage import binary_dilation, maximum_filter1d, mean, zoom
from scipy.spatial import cKDTree
from skimage import filters, measure
from skimage.segmentation import find_boundaries

import ncolor
import fastremap
import edt
from tqdm import trange
from dbscan import DBSCAN as new_DBSCAN

from .. import utils
from ..utils import get_module, Result
from ..gpu import torch_GPU, torch_CPU, ensure_torch, torch_and
from ..transforms.imports import normalize_field, normalize99, divergence
from ..transforms.imports import hysteresis_threshold as _hysteresis_threshold_torch

from ocdkit.array import torch_norm  # canonical gateway for ocdkit.array.torch_norm
from ocdkit.array.spatial import masks_to_affinity, boundary_to_masks  # canonical gateway for ocdkit.array.spatial ops
from ocdkit.array.union_find import cc_union_find as _cc_union_find

from ..logger import get_logger
core_logger = get_logger('core', color='#5c9edc')
