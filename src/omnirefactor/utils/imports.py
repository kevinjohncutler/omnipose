import io
import logging
import colorsys
import shutil
import tempfile
from urllib.request import urlopen

import numpy as np
import dask
from dask import array as da
from tqdm import tqdm
import cv2
import torch

from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    gaussian_filter,
    find_objects,
    generate_binary_structure,
    label,
    binary_fill_holes,
)
from scipy.ndimage import convolve1d, convolve, affine_transform
from scipy.spatial import ConvexHull
from sklearn.utils.extmath import cartesian

from skimage import color
from skimage.segmentation import find_boundaries

from mahotas.morph import hitmiss as mh_hitmiss
import math
import os
import re
from pathlib import Path

import mgen
import fastremap
import ncolor

import functools
import itertools

from ..logger import get_logger

omnipose_logger = get_logger('utils')

try:
    from skimage.morphology import remove_small_holes
    SKIMAGE_ENABLED = True
except ModuleNotFoundError:
    SKIMAGE_ENABLED = False

from ..gpu import ARM
