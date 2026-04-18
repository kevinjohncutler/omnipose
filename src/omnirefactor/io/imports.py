"""Centralized imports for the io subpackage (Layer 2).

Layer 2: depends on L0 (utils) and L1 (transforms) — never core or models.
"""

import os
import warnings
import glob
import re
from pathlib import Path
from csv import reader, writer
import shutil
import tempfile
from urllib.request import urlopen
from tqdm import tqdm

import numpy as np
from natsort import natsorted
import ncolor

from ..logger import get_logger
from .. import utils
from ..utils import Result
from ..transforms.imports import normalize99

from ocdkit.io import *  # canonical gateway for ocdkit.io
from ocdkit.morphology import masks_to_outlines  # canonical gateway for ocdkit.morphology

import matplotlib.pyplot as plt

io_logger = get_logger('io', color='#ff7f0e')
