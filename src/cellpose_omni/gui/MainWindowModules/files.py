import os
from .. import io
import numpy as np
from cellpose_omni.io import get_image_files

def dropEvent(self, event):
    files = [u.toLocalFile() for u in event.mimeData().urls()]
    if os.path.splitext(files[0])[-1] == '.npy':
        io._load_seg(self, filename=files[0])
    else:
        io._load_image(self, filename=files[0])
        
def dragEnterEvent(self, event):
    if event.mimeData().hasUrls():
        event.accept()
    else:
        event.ignore()

def get_files(self):
    folder = os.path.dirname(self.filename)
    mask_filter = '_masks'
    images = get_image_files(folder, mask_filter)
    fnames = [os.path.split(images[k])[-1] for k in range(len(images))]
    f0 = os.path.split(self.filename)[-1]
    idx = np.nonzero(np.array(fnames)==f0)[0][0]
    return images, idx

def get_prev_image(self):
    images, idx = self.get_files()
    idx = (idx-1)%len(images)
    io._load_image(self, filename=images[idx])

def get_next_image(self, load_seg=True):
    images, idx = self.get_files()
    idx = (idx+1)%len(images)
    io._load_image(self, filename=images[idx], load_seg=load_seg)
