"""Centralized imports for the metrics subpackage (Layer 3).

Layer 3: depends on L0 (utils), L2 (core, io) — never models.
"""

import numpy as np
import torch

from numba import jit
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import convolve, mean

from .. import utils, core
from ..utils import Result, kernel_setup, get_supporting_inds
from ..core.imports import torch_norm
from ..core.affinity import _get_affinity_torch
from ..io import masks_to_outlines

from torchvf.numerics import interp_vf, ivp_solver
