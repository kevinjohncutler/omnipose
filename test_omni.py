import pytest
import os
from pathlib import Path
import skimage.io


def test_omni():
    masks_dir = Path(os.path.dirname(ncolor.__file__)).parent.absolute()
    # masks = skimage.io.imread(os.path.join(masks_dir,'example.png'))
    