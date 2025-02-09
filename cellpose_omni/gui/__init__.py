
from cellpose_omni.io import check_dir

from omnipose.logger import setup_logger
logger = setup_logger('gui')


import signal, sys, os, pathlib, warnings, datetime, time
from ..utils import download_url_to_file, masks_to_outlines, diameters 


#logo 
ICON_PATH = pathlib.Path.home().joinpath('.omnipose','logo.png')
ICON_URL = 'https://github.com/kevinjohncutler/omnipose/blob/main/gui/logo.png?raw=true'


#test files
op_dir = pathlib.Path.home().joinpath('.omnipose','test_files')
check_dir(op_dir)
files = ['Sample000033.png','Sample000193.png','Sample000252.png','Sample000306.tiff','e1t1_crop.tif']
test_images = [pathlib.Path.home().joinpath(op_dir, f) for f in files]
for path,file in zip(test_images,files):
    if not path.is_file():
        download_url_to_file('https://github.com/kevinjohncutler/omnipose/blob/main/docs/test_files/'+file+'?raw=true',
                                path, progress=True)
PRELOAD_IMAGE = str(test_images[-1])
DEFAULT_MODEL = 'bact_phase_omni'



if not ICON_PATH.is_file():
    print('downloading logo from', ICON_URL,'to', ICON_PATH)
    download_url_to_file(ICON_URL, ICON_PATH, progress=True)

# Not everyone with have a math font installed, so all this effort just to have
# a cute little math-style gamma as a slider label...
GAMMA_PATH = pathlib.Path.home().joinpath('.omnipose','gamma.svg')
BRUSH_PATH = pathlib.Path.home().joinpath('.omnipose','brush.svg')

GAMMA_URL = 'https://github.com/kevinjohncutler/omnipose/blob/main/gui/gamma.svg?raw=true'   
if not GAMMA_PATH.is_file():
    print('downloading gamma icon from', GAMMA_URL,'to', GAMMA_PATH)
    download_url_to_file(GAMMA_URL, GAMMA_PATH, progress=True)
    