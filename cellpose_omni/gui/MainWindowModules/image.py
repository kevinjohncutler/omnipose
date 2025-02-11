from pyqtgraph import ViewBox
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSlot, QCoreApplication

from PyQt6.QtWidgets import QGraphicsPathItem
from PyQt6.QtGui import QPen, QBrush, QPainterPath, QTransform
from PyQt6.QtGui import QCursor
from PyQt6.QtCore import QPointF

from PyQt6 import QtGui, QtCore, QtWidgets

from .. import guiparts

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
        vals = self.slider.value()
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
    self.p0 = ViewBox(
        lockAspect=True,
        invertY=True,
        # border=pg.mkPen(color='red', width=1)
    )
    
    self.p0.setCursor(QtCore.Qt.CrossCursor)
    self.win.addItem(self.p0, 0, 0, rowspan=1, colspan=1)
    self.p0.setMenuEnabled(False)
    self.p0.setMouseEnabled(x=True, y=True)
    self.img = pg.ImageItem(viewbox=self.p0, parent=self,levels=(0,255))
    self.img.autoDownsample = False
    
    self.hist = guiparts.HistLUT(image=self.img,orientation='horizontal',gradientPosition='bottom')

    self.opacity_effect = QtWidgets.QGraphicsOpacityEffect()
    self.hist.setGraphicsEffect(self.opacity_effect)

    # self.win.addItem(self.hist,col=0,row=2)
    self.win.addItem(self.hist,col=0,row=1)

    # self.layer = guiparts.ImageDraw(viewbox=self.p0, parent=self)
    self.layer = guiparts.ImageDraw(parent=self)
    
    # self.scale = pg.ImageItem(viewbox=self.p0, parent=self,levels=(0,255))
    
    self.p0.scene().contextMenuItem = self.p0
    self.p0.addItem(self.img)
    self.p0.addItem(self.layer)
    # self.p0.addItem(self.scale)
    
    # self.p0.addItem(self.affinityOverlay)
    # self.vboOverlay = guiparts.GLLineOverlay(parent=self) # width=self.Lx, height=self.Ly, 
    # self.vboOverlay.setZValue(10)           # ensure it's on top
    # self.vboOverlay.setVisible(False)       # or True, default off
    # self.p0.addItem(self.vboOverlay)
        
    # self.l0.addWidget(self.affinityOverlay.get_widget())
    

                

    # Create the highlight path item - our cursor 
    self.highlight_rect = QGraphicsPathItem()
    # self.highlight_rect.setPen(Qt.PenStyle.NoPen) # no outline
    self.highlight_rect.setPen(QPen(Qt.PenStyle.NoPen))

    self.highlight_rect.setBrush(QBrush(pg.mkColor(255, 0, 0, 100)))  # Semi-transparent fill
    self.highlight_rect.setPen(pg.mkPen(color='red', width=1))

    self.p0.addItem(self.highlight_rect)

    # Add the rectangle to the ViewBox, which aligns it with the image coordinates
    self.p0.addItem(self.highlight_rect)

    # Connect mouse movement signal
    self.p0.scene().sigMouseMoved.connect(self.update_highlight)
    

def compute_saturation(self):
    # compute percentiles from stack
    self.saturation = []
    for n in range(len(self.stack)):
        # reverted for cellular images, maybe there can be an option?
        vals = self.slider.value()

        self.saturation.append([np.percentile(self.stack[n].astype(np.float32),vals[0]),
                                np.percentile(self.stack[n].astype(np.float32),vals[1])])
            


def recenter(self):
    buffer = 10 # leave some space between histogram and image
    dy = self.Ly+buffer
    dx = self.Lx
    
    # make room for scale disk
    # if self.ScaleOn.isChecked():
    #     dy += self.pr
        
    # set the range for whatever is the smallest dimension
    s = self.p0.screenGeometry()
    if s.width()>s.height():
        self.p0.setXRange(0,dx) #centers in x
        self.p0.setYRange(0,dy)
    else:
        self.p0.setYRange(0,dy) #centers in y
        self.p0.setXRange(0,dx)
        
    # unselect sector buttons
    self.quadbtns.setExclusive(False)
    for b in range(9):
        self.quadbtns.button(b).setChecked(False)      
    self.quadbtns.setExclusive(True)

