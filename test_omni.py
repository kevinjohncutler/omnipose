import pytest
import os
from pathlib import Path
import skimage.io
import omnipose

def test_omni():
    basedir = Path(os.path.dirname(omnipose.__file__)).parent.absolute()
    # masks = skimage.io.imread(os.path.join(masks_dir,'example.png'))
    
