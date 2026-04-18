"""Centralized imports for the models subpackage (Layer 4).

Layer 4: top-level API — may depend on all lower layers.
"""

import datetime
import os
from pathlib import Path
import time

import numpy as np
from tqdm.auto import tqdm, trange

import torch
from torch import nn, optim
from torch.amp import autocast, GradScaler
from contextlib import nullcontext

from scipy.ndimage import gaussian_filter
from scipy.stats import mode

import ncolor

from .. import core, data, transforms, metrics, utils, plot
from .registry import CP_MODELS, C2_MODEL_NAMES, BD_MODEL_NAMES, MODEL_NAMES
from ..gpu import empty_cache
from ..networks import assign_device, parse_model_string, UnetND
from ..transforms import torch_zoom
from ..transforms.filters import hysteresis_threshold
from ..core.steps import follow_flows_batch
from ..core.loss import loss as core_loss
from ..io import imread, check_dir
from ..data.eval import eval_set as EvalSet
from ..kwargs import split_kwargs, base_kwargs, split_kwargs_for

from .logging import models_logger, core_logger, tqdm_out

from ..utils import Result
