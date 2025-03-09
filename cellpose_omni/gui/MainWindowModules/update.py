from .. import logger # from __init__.py in parent directory
from omnipose.utils import normalize99
import numpy as np

from PyQt6.QtGui import QPalette
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QGraphicsItem
from PyQt6 import QtCore
import pyqtgraph as pg

class NonInteractiveHistogramLUTItem(pg.HistogramLUTItem):
    def event(self, event):
        # When in non-interactive mode, simply consume mouse and hover events.
        if event.type() in (QtCore.QEvent.Type.GraphicsSceneMousePress,
                            QtCore.QEvent.Type.GraphicsSceneMouseMove,
                            QtCore.QEvent.Type.GraphicsSceneMouseRelease,
                            QtCore.QEvent.Type.GraphicsSceneHoverEnter,
                            QtCore.QEvent.Type.GraphicsSceneHoverMove,
                            QtCore.QEvent.Type.GraphicsSceneHoverLeave):
            return True  # Consume the event; do nothing.
        return super().event(event)

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

def draw_change(self):
    if not self.SCheckBox.isChecked():
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
        
        vals = self.slider.value()
        image = normalize99(image,lower=vals[0],upper=vals[1])**self.gamma 
        # maybe should not directly modify image? Use viewer isntead?
        
        if self.invert.isChecked():
            image = 1-image
        
        # restore to uint8
        image *= 255

        # if self.color==0:
        #     self.img.setImage(image, autoLevels=False, lut=None)
        # elif self.color>0 and self.color<4:
        #     if not self.onechan:
        #         image = image[:,:,self.color-1]
        #     self.img.setImage(image, autoLevels=False, lut=self.cmap[self.color])
        # elif self.color==4:
        #     if not self.onechan:
        #         image = image.mean(axis=-1)
        #     self.img.setImage(image, autoLevels=False, lut=None)
        # elif self.color==5:
        #     if not self.onechan:
        #         image = image.mean(axis=-1)
        #     self.img.setImage(image, autoLevels=False, lut=self.cmap[0])
        
    else:
        image = np.zeros((self.Ly,self.Lx), np.uint8)
        if len(self.flows)>=self.view-1 and len(self.flows[self.view-1])>0:
            image = self.flows[self.view-1][self.currentZ]
    
            
        # if self.view==2: # distance
        #     # self.img.setImage(image,lut=pg.colormap.get('magma').getLookupTable(), levels=(0,255))
        #     self.img.setImage(image,autoLevels=False)
        # elif self.view==3: #boundary
        #     self.img.setImage(image,sutoLevels=False)
        # else:
        #     self.img.setImage(image, autoLevels=False, lut=None)
        # self.img.setLevels([0.0, 255.0])
        # self.set_hist()
    
    self.img.setImage(image,autoLevels=False)

    # Let users customize color maps and have them persist 
    state = self.states[self.view]
    if state is None: #should add a button to reset state to none and update plot
        self.hist.gradient.loadPreset(self.cmaps[self.view]) # select from predefined list
    else:
        self.hist.restoreState(state) #apply chosen color map
        
    self.set_hist_colors()
    
    # self.scale.setImage(self.radii, autoLevels=False)
    # self.scale.setLevels([0.0,255.0])
    #self.img.set_ColorMap(self.bwr)
    if self.NZ>1 and self.orthobtn.isChecked():
        self.update_ortho()
    
    # self.slider.setLow(self.saturation[self.currentZ][0])
    # self.slider.setHigh(self.saturation[self.currentZ][1])
    # if self.masksOn or self.outlinesOn:
    #     self.layer.setImage(self.layerz[self.currentZ], autoLevels=False) #<<< something to do with it 
    self.win.show()
    self.show()




def reset(self):
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
    # -- set menus to default -- #
    self.color = 0
    self.RGBDropDown.setCurrentIndex(self.color)
    self.view = 0
    self.RGBChoose.button(self.view).setChecked(True)
    self.SCheckBox.setChecked(True)
    # self.SCheckBox.setEnabled(False)
    self.restore_masks = 0
    self.states = [None for i in range(len(self.default_cmaps))] 

    # -- zero out image stack -- #
    self.opacity = 128 # how opaque masks should be
    self.outcolor = np.array([1,0,0,.5])*255
    self.NZ, self.Ly, self.Lx = 1,512,512
    self.saturation = [[0,255] for n in range(self.NZ)]
    self.gamma = 1
    self.slider.setMinimum(0)
    self.slider.setMaximum(100)
    self.slider.show()
    self.currentZ = 0
    self.flows = [[],[],[],[],[[]]]
    self.stack = np.zeros((1,self.Ly,self.Lx,3))
    # masks matrix
    self.layerz = np.zeros((self.Ly,self.Lx,4), np.uint8)
    # image matrix with a scale disk
    self.radii = 0*np.ones((self.Ly,self.Lx,4), np.uint8)
    
    self.cellpix = np.zeros((1,self.Ly,self.Lx), np.uint32)
    self.outpix = np.zeros((1,self.Ly,self.Lx), np.uint32)
    
    self.masks = np.zeros((self.Ly,self.Lx), np.uint32)
    self.bounds = np.zeros((self.Ly,self.Lx), np.uint32)
    
    self.links = None
    
    self.initialize_seg()
    # print('reset',self.outpix.shape,self.affinity_graph.shape)
    
    self.ismanual = np.zeros(0, 'bool')
    self.accent = self.palette().brush(QPalette.ColorRole.Highlight).color()
    self.update_plot()
    self.progress.setValue(0)
    self.orthobtn.setChecked(False)
    self.filename = []
    self.loaded = False
    self.recompute_masks = False
    