from PyQt6.QtGui import QPen, QBrush, QPainterPath, QTransform
import numpy as np

def toggle_removals(self):
    if self.ncells>0:
        self.ClearButton.setEnabled(True)
        # self.remcell.setEnabled(True)
        self.undo.setEnabled(True)
    else:
        self.ClearButton.setEnabled(False)
        # self.remcell.setEnabled(False)
        self.undo.setEnabled(False)

def toggle_mask_ops(self):
    self.toggle_removals()


def compute_kernel_path(self, kernel):
    path = QPainterPath()
    for ky, kx in np.argwhere(kernel == 1):
        path.addRect(kx, ky, 1, 1)  # Create the kernel path
    self.highlight_path = path

def clear_all(self):
    self.save_state()
    self.prev_selected = 0
    self.selected = 0
    self.layerz = np.zeros((self.Ly,self.Lx,4), np.uint8)
    self.mask_stack = np.zeros((self.NZ,self.Ly,self.Lx), np.uint32)
    self.outl_stack = np.zeros((self.NZ,self.Ly,self.Lx), np.uint32)
    # self.cellcolors = np.array([255,255,255])[np.newaxis,:]
    self.ncells = 0
    # self.toggle_removals()
    self.update_layer()
