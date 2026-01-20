
from cellpose_omni.io import check_dir

from omnipose.logger import setup_logger
logger = setup_logger('gui')


import signal, sys, os, pathlib, warnings, datetime, time
from importlib.resources import files as resource_files
from ..utils import download_url_to_file, masks_to_outlines, diameters 


# logo assets (packaged with the wheel)
ASSETS_DIR = resource_files("cellpose_omni.gui.assets")
ICON_PATH = ASSETS_DIR / "logo.png"


# test files
op_dir = pathlib.Path.home().joinpath('.omnipose','test_files')
check_dir(op_dir)
files = ['Sample000033.png','Sample000193.png','Sample000252.png','Sample000306.tiff','e1t1_crop.tif']
test_images = [pathlib.Path.home().joinpath(op_dir, f) for f in files]
for path,file in zip(test_images,files):
    if not path.is_file():
        download_url_to_file('https://github.com/kevinjohncutler/omnipose/blob/main/docs/test_files/'+file+'?raw=true',
                                path, progress=True)
PRELOAD_IMAGE = str(test_images[-1])
DEFAULT_MODEL = 'bact_phase_affinity'



# Not everyone will have a math font installed, so use packaged SVG assets.
GAMMA_PATH = ASSETS_DIR / "gamma.svg"
BRUSH_PATH = ASSETS_DIR / "brush.svg"
    
