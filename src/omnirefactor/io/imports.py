import os
import datetime
import gc
import warnings
import glob
import logging
import pathlib
import sys
import re
from pathlib import Path
from csv import reader, writer
import shutil
import tempfile
import subprocess
from urllib.request import urlopen
from tqdm import tqdm

import numpy as np
import cv2
import tifffile
from natsort import natsorted
from aicsimageio import AICSImage
import ncolor

from ..logger import LOGGER_FORMAT, get_logger
from .. import utils, plot, transforms

try:
    from PyQt6 import QtGui, QtCore, QtWidgets
    GUI = True
except Exception:
    GUI = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB = True
except Exception:
    MATPLOTLIB = False

# try:
#     from google.cloud import storage
#     SERVER_UPLOAD = True
# except:
SERVER_UPLOAD = False

io_logger = get_logger('io', color='#ff7f0e')
