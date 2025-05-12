from .. import logger # from __init__.py in parent directory
from omnipose.utils import normalize99
import numpy as np

from PyQt6.QtGui import QPalette, QCursor
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QGraphicsItem
from PyQt6 import QtCore
import pyqtgraph as pg
from PyQt6 import QtGui


def update_roi_count(self):
    self.roi_count.setText(f'{self.ncells} RoIs')        

def update_shape(self): 
    self.Ly, self.Lx, _ = self.stack[self.currentZ].shape
    self.shape = (self.Ly, self.Lx)
    # self.pixelGridOverlay.update_size(self.Lx, self.Ly)
    
def update_layer(self):
    logger.info(f'updating layer {self.loaded}')
    self.draw_layer()
    self.update_roi_count()
    self.win.show()
    self.show()
    
def update_layer_and_graph(self):
    self.update_layer()
    # reinit overlay item 
    # torn on whether to just include this in update_layer
    # it's not limiting performance to have it there, but usually not necessary 
    if hasattr(self, 'pixelGridOverlay'):
        logger.info(f'resetting pixel grid')
        self.pixelGridOverlay.reset()
    else:
        logger.info(f'no pixelGridOverlay to reset')
        

def draw_change(self):
    if not self.PencilCheckBox.isChecked():
        self.highlight_rect.hide()
    else:
        self.update_highlight()

def update_plot(self):
    self.update_shape()
    
    # toggle off histogram for flow field 
    if self.view == 1:
        self.opacity_effect.setOpacity(0.0)
        self.hist.show_histogram = False
    else:
        self.opacity_effect.setOpacity(1.0)
        self.hist.show_histogram = True

    if self.NZ < 2:
        self.scroll.hide()
    else:
        self.scroll.show()
            
    if self.view==0:
        # self.hist.restoreState(self.histmap_img)
        image = self.stack[self.currentZ]
        if self.onechan:
            # show single channel
            image = self.stack[self.currentZ,:,:,0]
        
        vals = self.contrast_slider.value()
        image = normalize99(image,lower=vals[0],upper=vals[1])**self.gamma 
        # maybe should not directly modify image? Use viewer instead?
        
        if self.invert.isChecked():
            image = 1-image

    elif self.view==4:
        image = self.csum.astype(np.float32)# will need to generalize to Z
    else:
        image = np.zeros((self.Ly,self.Lx), np.uint8)
        if len(self.flows)>=self.view-2 and len(self.flows[self.view-1])>0:
            image = self.flows[self.view-1][self.currentZ] #/ 255 
    

    self.hist.set_view(
        v=self.view,
        preset=self.cmaps[self.view],
        default_cmaps=self.default_cmaps
    )
    
    # Set the histogram levels for the image.
    dmin, dmax = float(image.min()), float(image.max())
    if dmax == dmin:
        dmax = dmin + 1.0
    levels = (dmin, dmax)
    lut = None

        
    if self.view==4:     
        nBands = int(dmax+1)  # assuming image values are integers 0..(nBands-1)
        self.hist.setDiscreteMode(True, n_bands=nBands)  # this overrides the gradient painting
        # Force image levels to exactly match discrete bands (e.g. 0 to nBands-1)
        levels=(0, nBands-1)
        
    else:
        self.hist.setDiscreteMode(False)  # restore continuous mode

    # print('ttt',levels, image.ndim, image.shape, lut)
    
    # Decide whether to treat it as grayscale or color:
    if image.ndim == 3 and image.shape[-1] == 1:
        # Single-channel (H, W, 1) => reshape to (H, W) so we can apply LUT.
        image = image[..., 0]
        self.img.setImage(image, autoLevels=False, levels=levels)
    elif image.ndim == 3 and image.shape[-1] in (3, 4):
        # Multi-channel (RGB or RGBA): show color image, no LUT.
        self.img.setImage(image, autoLevels=False, levels=levels, lut=lut)
    else:
        # Otherwise, assume it's already 2D or something else.
        self.img.setImage(image, autoLevels=False, levels=levels)


    self.set_hist_colors()
    
    if self.NZ>1 and self.orthobtn.isChecked():
        self.update_ortho()
    

    self.win.show()
    self.show()

def reset(self):
    logger.info(f'mainwindow reset() called')

    # ---- start sets of points ---- #
    self.selected = 0
    self.X2 = 0
    # self.resize = -1
    self.onechan = False
    self.loaded = False
    self.channel = [0,1]
    self.current_point_set = []
    self.in_stroke = False
    self.strokes = []
    self.stroke_appended = True
    self.ncells = 0
    self.zdraw = []
    self.removed_cell = []
    self.cellcolors = np.array([255,255,255])[np.newaxis,:]
    self.ncellcolors = 0
    
    # -- set menus to default -- #
    self.color = 0
    self.RGBDropDown.setCurrentIndex(self.color)
    self.view = 0
    self.ViewChoose.button(self.view).setChecked(True)
    self.PencilCheckBox.setChecked(False)
    # self.PencilCheckBox.setEnabled(False)
    self.restore_masks = 0
    # self.states = [None for i in range(len(self.default_cmaps))] 
    # if not hasattr(self.hist.gradient, 'view_states'):
    #     self.hist.gradient.view_states = {}
    

    # -- zero out image stack -- #
    self.opacity = 128 # how opaque masks should be
    self.outcolor = np.array([1,0,0,.5])*255
    
    if getattr(self, 'stack', None) is None:
        self.NZ, self.Ly, self.Lx = 1,512,512
        self.stack = np.zeros((1,self.Ly,self.Lx,3))
        
    else:
        shape = self.stack.shape # assume ZYXC here 
        logger.info(f'resetting with ZYXC stack, shape: {shape}')
        self.Ly, self.Lx = shape[1:3]
        self.NZ = shape[0]
        

    
    self.saturation = [[0,255] for n in range(self.NZ)]
    self.gamma = 1
    self.contrast_slider.setMinimum(0)
    self.contrast_slider.setMaximum(100)
    self.contrast_slider.show()
    self.currentZ = 0
    self.flows = [[],[],[],[],[[]]]
    
    # masks matrix
    self.layerz = np.zeros((self.Ly,self.Lx,4), np.uint8)
    # image matrix with a scale disk
    self.radii = 0*np.ones((self.Ly,self.Lx,4), np.uint8)
    
    self.mask_stack = np.zeros((self.NZ,self.Ly,self.Lx), np.uint32)
    self.outl_stack = np.zeros((self.NZ,self.Ly,self.Lx), np.uint32)
    
    self.masks = np.zeros((self.Ly,self.Lx), np.uint32)
    self.bounds = np.zeros((self.Ly,self.Lx), np.uint32)
    
    self.links = None
    
    #should reset affinity graph here too 
    self.initialize_seg(compute_affinity=True) # compute_affinity=True means reset affinity

    # self.recenter()
    
    # reinit overlay item 
    if hasattr(self, 'pixelGridOverlay'):
        logger.info(f'resetting pixel grid')
        self.pixelGridOverlay.reset()
    else:
        logger.info(f'no pixelGridOverlay to reset')
        
    
    self.ismanual = np.zeros(0, 'bool')
    self.accent = self.palette().brush(QPalette.ColorRole.Highlight).color()
    self.update_plot()
    self.progress.setValue(0)
    self.orthobtn.setChecked(False)
    self.filename = []
    self.loaded = False
    self.recompute_masks = False

def eventFilter(self, obj, event):
    # Filter events only for the viewport, ignoring sliders/other widgets.
    if obj != self.win.viewport():
        return False

    # Print debug info if needed.
    # print('eventFilter', obj, event.type())

    if event.type() == QtCore.QEvent.Type.Leave:
        self.highlight_rect.hide()
        return True

    elif event.type() == QtCore.QEvent.Type.MouseMove:
        widget_pos = self.win.viewport().mapFromGlobal(QCursor.pos())
        if not self.win.viewport().rect().contains(widget_pos):
            self.highlight_rect.hide()
            return True

    return False