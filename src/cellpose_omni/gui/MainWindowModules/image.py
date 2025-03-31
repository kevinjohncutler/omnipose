from pyqtgraph import ViewBox
import pyqtgraph as pg

from PyQt6.QtWidgets import QGraphicsPathItem

from PyQt6 import QtGui, QtCore, QtWidgets

from .. import guiparts
from .. import logger

import numpy as np

def move_in_Z(self):
    if self.loaded:
        self.currentZ = min(self.NZ, max(0, int(self.scroll.value())))
        self.zpos.setText(str(self.currentZ))
        self.update_plot()
        self.draw_layer()
        self.update_layer()
        
def update_ztext(self):
    zpos = self.currentZ
    try:
        zpos = int(self.zpos.text())
    except:
        logger.warning('ERROR: zposition is not a number')
    self.currentZ = max(0, min(self.NZ-1, zpos))
    self.zpos.setText(str(self.currentZ))
    self.scroll.setValue(self.currentZ)

def level_change(self):
    if self.loaded:
        vals = self.contrast_slider.value()
        self.ops_plot = {'saturation': vals}
        self.saturation[self.currentZ] = vals
        self.update_plot()

def gamma_change(self):
    if self.loaded:
        val = self.gamma_slider.value()
        self.ops_plot = {'gamma': val}
        # self.gamma[self.currentZ] = val
        self.gamma = val
        self.update_plot()
        
    # for viewbox code below
    # policy = QtWidgets.QSizePolicy()
    # policy.setRetainSizeWhenHidden(True)
    # self.hist.setSizePolicy(policy)
                # Add a highlight rectangle for kernel preview
    # self.kernel_size = (1,1)  # Size of the kernel in pixels
    # self.highlight_rect = pg.QtWidgets.QGraphicsRectItem()
    # self.highlight_rect.setPen(pg.mkPen(color='red', width=1))
    # self.highlight_rect.setBrush(pg.mkBrush(color=(255, 0, 0, 50)))  # Semi-transparent fill 

def make_viewbox(self):
    self.viewbox = ViewBox(
        lockAspect=True,
        invertY=True,
        # border=pg.mkPen(color='red', width=1)
    )
    
    self.viewbox.setCursor(QtCore.Qt.CrossCursor)
    self.win.addItem(self.viewbox, 0, 0, rowspan=1, colspan=1)
    self.viewbox.setMenuEnabled(False)
    self.viewbox.setMouseEnabled(x=True, y=True)
    self.img = pg.ImageItem(viewbox=self.viewbox, parent=self,levels=(0,255))
    self.img.autoDownsample = False
    
    self.hist = guiparts.HistLUT(image=self.img,orientation='horizontal',gradientPosition='bottom')
    self.gradient = self.hist.gradient
    self.opacity_effect = QtWidgets.QGraphicsOpacityEffect()
    self.hist.setGraphicsEffect(self.opacity_effect)
    self.win.addItem(self.hist,col=0,row=1)

    self.layer = guiparts.ImageDraw(parent=self)
    
    self.viewbox.scene().contextMenuItem = self.viewbox
    self.viewbox.addItem(self.img)
    self.viewbox.addItem(self.layer)
    
    # Create the highlight cursor
    self.highlight_rect = QGraphicsPathItem()
    self.viewbox.addItem(self.highlight_rect)

    # Connect mouse movement signal
    self.viewbox.scene().sigMouseMoved.connect(self.update_highlight)

def compute_saturation(self):
    # compute percentiles from stack
    self.saturation = []
    for n in range(len(self.stack)):
        # reverted for cellular images, maybe there can be an option?
        vals = self.contrast_slider.value()

        self.saturation.append([np.percentile(self.stack[n].astype(np.float32),vals[0]),
                                np.percentile(self.stack[n].astype(np.float32),vals[1])])
            

def recenter(self):
    # Temporarily unlock the aspect ratio so autoRange can fit everything
    self.viewbox.setAspectLocked(False)

    # Re-center and fit to the entire image
    self.viewbox.autoRange(items=[self.img], padding=0.05)

    # Re-lock aspect ratio if you want a square-ish zoom behavior
    self.viewbox.setAspectLocked(True)

    # Unselect sector buttons
    self.quadbtns.setExclusive(False)
    for b in range(9):
        self.quadbtns.button(b).setChecked(False)
    self.quadbtns.setExclusive(True)
    

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
