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

from ocdkit.io import *
from ocdkit.array import normalize99
from ocdkit.result import Result
from ocdkit.morphology import masks_to_outlines

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB = True
except Exception:
    MATPLOTLIB = False

io_logger = get_logger('io', color='#ff7f0e')
