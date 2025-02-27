from PyQt6 import QtGui, QtCore
from PyQt6.QtGui import QPainter
from PyQt6.QtWidgets import QApplication, QRadioButton, QWidget, QDialog, QButtonGroup, QSlider, QStyle, QStyleOptionSlider, QGridLayout, QPushButton, QLabel, QLineEdit, QDialogButtonBox, QComboBox
import pyqtgraph as pg
from pyqtgraph import Point
import numpy as np
import os
from omnipose.core import affinity_to_boundary

from omnipose import core

# import superqt

TOOLBAR_WIDTH = 7
SPACING = 3
WIDTH_0 = 25

from PyQt6.QtWidgets import QPlainTextEdit, QFrame

from PyQt6.QtCore import QObject, QEvent

class NoMouseFilter(QObject):
    def eventFilter(self, obj, event):
        if event.type() in (QEvent.Type.GraphicsSceneMousePress,
                            QEvent.Type.GraphicsSceneMouseMove,
                            QEvent.Type.GraphicsSceneMouseRelease):
            return True  # Consume the event.
        return super().eventFilter(obj, event)

class TextField(QPlainTextEdit):
    clicked= QtCore.pyqtSignal()
    def __init__(self,widget,parent=None):
        super().__init__(widget)
        # self.setStyleSheet(self.parent().textbox_style)
    def mousePressEvent(self,QMouseEvent):
        self.clicked.emit()

class QHLine(QFrame):
    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(QFrame.Shape.HLine)
        # self.setFrameShadow(QFrame.Shape.Sunken)


def create_channel_choose():
    # choose channel
    ChannelChoose = [QComboBox(), QComboBox()]
    ChannelLabels = []
    ChannelChoose[0].addItems(['gray','red','green','blue'])
    ChannelChoose[1].addItems(['none','red','green','blue'])
    cstr = ['chan to segment:', 'chan2 (optional): ']
    for i in range(2):
        ChannelLabels.append(QLabel(cstr[i]))
        if i==0:
            ChannelLabels[i].setToolTip('this is the channel in which the cytoplasm or nuclei exist that you want to segment')
            ChannelChoose[i].setToolTip('this is the channel in which the cytoplasm or nuclei exist that you want to segment')
        else:
            ChannelLabels[i].setToolTip('if <em>cytoplasm</em> model is chosen, and you also have a nuclear channel, then choose the nuclear channel for this option')
            ChannelChoose[i].setToolTip('if <em>cytoplasm</em> model is chosen, and you also have a nuclear channel, then choose the nuclear channel for this option')
        
    return ChannelChoose, ChannelLabels

class ModelButton(QPushButton):
    def __init__(self, parent, model_name, text):
        super().__init__()
        self.setEnabled(False)
        self.setText(text)
        self.setFont(parent.smallfont)
        self.clicked.connect(lambda: self.press(parent))
        self.model_name = model_name
        
    def press(self, parent):
        parent.compute_model(self.model_name)

class TrainWindow(QDialog):
    def __init__(self, parent, model_strings):
        super().__init__(parent)
        self.setGeometry(100,100,900,350)
        self.setWindowTitle('train settings')
        self.win = QWidget(self)
        self.l0 = QGridLayout()
        self.win.setLayout(self.l0)

        yoff = 0
        qlabel = QLabel('train model w/ images + _seg.npy in current folder >>')
        qlabel.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Weight.Bold))
        
        qlabel.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        self.l0.addWidget(qlabel, yoff,0,1,2)

        # choose initial model
        yoff+=1
        self.ModelChoose = QComboBox()
        self.ModelChoose.addItems(model_strings)
        self.ModelChoose.addItems(['scratch']) 
        self.ModelChoose.setFixedWidth(150)
        self.ModelChoose.setCurrentIndex(parent.training_params['model_index'])
        self.l0.addWidget(self.ModelChoose, yoff, 1,1,1)
        qlabel = QLabel('initial model: ')
        qlabel.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.l0.addWidget(qlabel, yoff,0,1,1)

        # choose channels
        self.ChannelChoose, self.ChannelLabels = create_channel_choose()
        for i in range(2):
            yoff+=1
            self.ChannelChoose[i].setFixedWidth(150)
            self.ChannelChoose[i].setCurrentIndex(parent.ChannelChoose[i].currentIndex())
            self.l0.addWidget(self.ChannelLabels[i], yoff, 0,1,1)
            self.l0.addWidget(self.ChannelChoose[i], yoff, 1,1,1)

        # choose parameters        
        labels = ['learning_rate', 'weight_decay', 'n_epochs', 'model_name']
        self.edits = []
        yoff += 1
        for i, label in enumerate(labels):
            qlabel = QLabel(label)
            qlabel.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.l0.addWidget(qlabel, i+yoff,0,1,1)
            self.edits.append(QLineEdit())
            self.edits[-1].setText(str(parent.training_params[label]))
            self.edits[-1].setFixedWidth(200)
            self.l0.addWidget(self.edits[-1], i+yoff, 1,1,1)

        yoff+=len(labels)

        yoff+=1
        qlabel = QLabel('(to remove files, click cancel then remove \nfrom folder and reopen train window)')
        self.l0.addWidget(qlabel, yoff,0,2,4)

        # click button
        yoff+=2
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(lambda: self.accept(parent))
        self.buttonBox.rejected.connect(self.reject)
        self.l0.addWidget(self.buttonBox, yoff, 0, 1,4)

        # list files in folder
        qlabel = QLabel('filenames')
        qlabel.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Weight.Bold))
        self.l0.addWidget(qlabel, 0,4,1,1)
        qlabel = QLabel('# of masks')
        qlabel.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Weight.Bold))
        self.l0.addWidget(qlabel, 0,5,1,1)
    
        for i in range(10):
            if i > len(parent.train_files) - 1:
                break
            elif i==9 and len(parent.train_files) > 10:
                label = '...'
                nmasks = '...'
            else:
                label = os.path.split(parent.train_files[i])[-1]
                nmasks = str(parent.train_labels[i].max())
            qlabel = QLabel(label)
            self.l0.addWidget(qlabel, i+1,4,1,1)
            qlabel = QLabel(nmasks)
            qlabel.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.l0.addWidget(qlabel, i+1, 5,1,1)

    def accept(self, parent):
        # set channels
        for i in range(2):
            parent.ChannelChoose[i].setCurrentIndex(self.ChannelChoose[i].currentIndex())
        # set training params
        parent.training_params = {'model_index': self.ModelChoose.currentIndex(),
                                 'learning_rate': float(self.edits[0].text()), 
                                 'weight_decay': float(self.edits[1].text()), 
                                 'n_epochs':  int(self.edits[2].text()),
                                 'model_name': self.edits[3].text()
                                 }
        self.done(1)

def make_quadrants(parent, yp):
    """ make quadrant buttons """
    parent.quadbtns = QButtonGroup(parent)
    for b in range(9):
        btn = QuadButton(b, ''+str(b+1), parent)
        parent.quadbtns.addButton(btn, b)
        parent.l0.addWidget(btn, yp + parent.quadbtns.button(b).ypos, TOOLBAR_WIDTH-3+parent.quadbtns.button(b).xpos, 1, 1)
        btn.setEnabled(True)
    parent.quadbtns.setExclusive(True) # this makes the loop below unneeded 

class QuadButton(QPushButton):
    """ custom QPushButton class for quadrant plotting
        requires buttons to put into a QButtonGroup (parent.quadbtns)
         allows only 1 button to pressed at a time
    """
    def __init__(self, bid, Text, parent=None):
        super(QuadButton,self).__init__(parent)
        self.setText(Text)
        self.setCheckable(True)
        # self.setStyleSheet(parent.styleUnpressed)
        # self.setFont(QtGui.QFont("Arial", 10))
        # self.resize(self.minimumSizeHint())
        self.setFixedWidth(WIDTH_0)
        self.setFixedHeight(WIDTH_0)
        self.xpos = bid%3
        self.ypos = int(np.floor(bid/3))
        self.clicked.connect(lambda: self.press(parent, bid))
        self.show()

    def press(self, parent, bid):
        # for b in range(9):
        #     if parent.quadbtns.button(b).isEnabled():
        #         # parent.quadbtns.button(b).setStyleSheet(parent.styleUnpressed)
        #         parent.quadbtns.button(b).setChecked(False)


        # self.setStyleSheet(parent.stylePressed)
        
        self.xrange = np.array([self.xpos-.2, self.xpos+1.2]) * parent.Lx/3
        self.yrange = np.array([self.ypos-.2, self.ypos+1.2]) * parent.Ly/3
        # change the zoom
        parent.p0.setXRange(self.xrange[0], self.xrange[1])
        parent.p0.setYRange(self.yrange[0], self.yrange[1])
        parent.show()



class TypeRadioButtons(QButtonGroup):
    def __init__(self, parent=None, row=0, col=0):
        super(TypeRadioButtons, self).__init__()
        parent.color = 0
        self.parent = parent
        self.bstr = self.parent.cell_types
        for b in range(len(self.bstr)):
            button = QRadioButton(self.bstr[b])
            # button.setStyleSheet('color: rgb(190,190,190);')
            button.setFont(QtGui.QFont("Arial", 10))
            if b==0:
                button.setChecked(True)
            self.addButton(button, b)
            button.toggled.connect(lambda: self.btnpress(parent))
            self.parent.l0.addWidget(button, row+b,col,1,2)
        self.setExclusive(True)
        #self.buttons.

    def btnpress(self, parent):
        b = self.checkedId()
        self.parent.cell_type = b

class RGBRadioButtons(QButtonGroup):
    def __init__(self, parent=None, row=0, col=0):
        super(RGBRadioButtons, self).__init__()
        parent.color = 0
        self.parent = parent
        self.bstr = ["image", "flow field", "distance field", "boundary logits"]
        #self.buttons = QButtonGroup()
        self.dropdown = []
        for b in range(len(self.bstr)):
            button = QRadioButton(self.bstr[b])
            # button.setStyleSheet('color: white;')
            button.setFont(parent.medfont)
            if b==0:
                button.setChecked(True)
            self.addButton(button, b)
            button.clicked.connect(lambda: self.btnpress(parent)) #"toggled" sends update twice
            self.parent.l0.addWidget(button, row+b,col,1,2)
        # self.setExclusive(True)
        #self.buttons.

    def btnpress(self, parent):
        b = self.checkedId()
        self.parent.view = b
        if self.parent.loaded:
            self.parent.update_layer()
            self.parent.update_plot()


class ImageDraw(pg.ImageItem):
    def __init__(self, data=None, parent=None):
        """
        data: 2D or 3D (H x W x RGBA) numpy array for the layer you want to paint on.
        parent: any object that has:
            - brush_size : int
            - spacePressed : bool
        """
        super().__init__()
        self.parent = parent
        self.parent.current_label = 0
        
        # Store the underlying layer data; display it
        if data is None:
            data = np.zeros((512, 512, 4), dtype=np.uint8)  # Dummy fallback
        self._data = data
        self.setImage(self._data)

        self._drawing = False
        self._lastPos = None

    def mousePressEvent(self, event):
    
        x, y = int(event.pos().x()), int(event.pos().y())

        # Safely check for pick_label_enabled and flood_fill_enabled
        if getattr(self.parent, 'pick_label_enabled', False):
            self._pickLabel(x, y)
            event.accept()
            
        elif getattr(self.parent, 'flood_fill_enabled', False):
            self.parent.save_state()
            self._floodFill(x, y, getattr(self.parent, 'current_label', 0))  # Default label to 0
            event.accept()
            
        elif self._canDraw(event):
            self.parent.save_state()
            self._drawing = True
            self._lastPos = event.pos()
            self._drawPixel(x, y) 
            # self._paintLine(self._lastPos, event.pos()) # test
            event.accept()
        else:
            pg.ImageItem.mousePressEvent(self, event)
            
    def mouseMoveEvent(self, event):
        if self._drawing:
            self._paintLine(self._lastPos, event.pos())
            self._lastPos = event.pos()
            event.accept()
        else:
            pg.ImageItem.mouseMoveEvent(self, event)
            

    def mouseReleaseEvent(self, event):
        if self._drawing:
            # self._paintLine(self._lastPos, event.pos())
            x, y = int(event.pos().x()), int(event.pos().y())
            self._drawPixel(x, y, cleanup=True) 
            # self._cleanup()
            self._drawing = False
            self._lastPos = None
            self.parent.redo_stack.clear()
            self.parent.update_layer() # is this needed? <<<<<<<<<<<<<<<<
            event.accept()
        else:
            # super().mouseReleaseEvent(event)
            pg.ImageItem.mouseReleaseEvent(self, event)
            
    def _canDraw(self, event):
        """Checks if conditions allow drawing instead of panning."""
        # If brush_size is 0 or space is pressed, do not draw
        if not self.parent.SCheckBox.isChecked():
            return False
        if getattr(self.parent, 'spacePressed', False):
            return False
        # Must be left mouse
        if event.button() != QtCore.Qt.LeftButton and not (event.buttons() & QtCore.Qt.LeftButton):
            return False
        return True

    # def _paintAt(self, pos):
    #     """Draw a single point (useful for initial press)."""
    #     self._drawPixel(pos.x(), pos.y())

    def _paintLine(self, startPos, endPos):
        """Draw a line from start to end by interpolating points."""
        if startPos is None or endPos is None:
            return
        x0, y0 = int(startPos.x()), int(startPos.y())
        x1, y1 = int(endPos.x()),   int(endPos.y())

        # Simple linear interpolation or Bresenham’s line. Here’s a quick approach:
        num_steps = max(abs(x1 - x0), abs(y1 - y0)) + 1
        xs = np.linspace(x0, x1, num_steps, dtype=int)
        ys = np.linspace(y0, y1, num_steps, dtype=int)
        
        for x, y in zip(xs, ys):
            self._drawPixel(x, y)

    def _drawPixel(self, x, y, cleanup=False):
        """Draws a circular area or single pixel using a precomputed kernel."""
        
        brush_size = getattr(self.parent, 'brush_size', 1)
        label = getattr(self.parent, 'current_label', 0)
        z = self.parent.currentZ
        masks = self.parent.cellpix[z]
        bounds = self.parent.outpix[z] # this might be a direct hook to the display
        
        affinity = self.parent.affinity_graph
        height, width = masks.shape

        # Determine the radius of the kernel
        kr = 0 if brush_size == 1 else brush_size // 2
        if not hasattr(self, '_kernel') or self._kernel.shape[0] != 2 * kr + 1:
            self._generateKernel(brush_size)

        # Array slice (r - row, c - column)
        r0, r1 = max(0, y - kr), min(y + kr + 1, height)
        c0, c1 = max(0, x - kr), min(x + kr + 1, width)
        arr_slc = (slice(r0, r1), slice(c0, c1))

        # Kernel slice
        kernel = self._kernel
        kr0, kr1 = max(0, kr - y), kr + (r1 - y)
        kc0, kc1 = max(0, kr - x), kr + (c1 - x)
        ker_slc = (slice(kr0, kr1), slice(kc0, kc1))
        
        # Dilated slice (expand by 1 in all directions)
        d = 3
        dil_r0, dil_r1 = max(0, r0 - d), min(r1 + d, height)
        dil_c0, dil_c1 = max(0, c0 - d), min(c1 + d, width)
        dil_slc = (slice(dil_r0, dil_r1), slice(dil_c0, dil_c1))

        # similar to _get_affinity, should check that too for some factoring out
        # source inds restricts to valid sources 
        source_indices = self.parent.ind_matrix[arr_slc][kernel[ker_slc]]
        # source_indices = source_indices[source_indices>-1]
        source_coords = tuple(c[source_indices] for c in self.parent.coords)
        targets = []
        
        # print('fff\n', self.parent.coords[0].shape, np.prod(masks.shape), '\n')
        
        masks[source_coords] = label # apply it here
        
        if np.any(source_indices==-1):
            print('\n ERROR, negative index in source_indices')

        steps = self.parent.steps
        dim = self.parent.dim
        idx = self.parent.inds[0][0]
        # print('affinity none?', affinity is None)
        # print('affinity shape', affinity.shape, np.any(affinity))
        

        if affinity is not None and affinity.shape[-dim:] == masks.shape:
            for i in self.parent.non_self:
                step = steps[i]
                target_coords = tuple(c+s for c,s in zip(source_coords, step))
                
               # source_labels = masks[source_coords]
                target_labels = masks[target_coords]
                if label!=0:
                    connect = target_labels==label
                else:
                    connect = False
                
                # update affinity graph
                affinity[i][source_coords] = affinity[-(i+1)][target_coords] = connect
                # affinity[i][target_coords] = affinity[-(i+1)][source_coords] = connect
                
                targets.append(target_coords)
                
        else:
            print('affinity graph not initialized') 
               
    
        # define a region around the source and target pixels
        targets.append(source_coords)
        surround_coords = tuple(np.concatenate(arrays, axis=0) for arrays in zip(*targets))

        

# !!! while drawing, could enable a mode not to connect to self unless it is to the direction opposite motion, so tht we can draw disconnected snakes 


        
        update_inds = []
        update_alpha = []
        # have to wait to update affinity after looping over all directions   
        for i in self.parent.non_self[:idx]:
            step = steps[i]
            target_coords = tuple(c+s for c,s in zip(source_coords, step))
            
            inds = self.parent.pixelGridOverlay.lineIndices[source_coords +(i,)].tolist() # need 3.11 for this syntax?
            
            opp_target_coords = tuple(c-s for c,s in zip(source_coords, step))
            # inds += self.parent.pixelGridOverlay.lineIndices[*opp_target_coords,i].tolist()
            inds += self.parent.pixelGridOverlay.lineIndices[opp_target_coords+(i,)].tolist()
            

            update_inds += inds
            update_alpha.append(affinity[i][source_coords])
            update_alpha.append(affinity[i][opp_target_coords])
                    
        self.parent.pixelGridOverlay.hide_lines_batch(update_inds, 
                                                      np.concatenate(update_alpha), 
                                                      visible=False)
    
                
        # I think I am missing the background pixels connected to it
        # maybe the cleanup will do it 

        # I could add a cleanup step here from get_affinity_torch
        # it could operate on just the dilated region to update the affinity graph
                 
    

        
        # some strangeness with outlines on and masks off, as if the masks are zero?
                    
        # update boundary

        # print('\n\nA\n\n', affinity.shape, masks.shape,  masks[surround_coords].shape, affinity[:,*surround_coords].shape)
        # print('surround_coords', surround_coords,targets)
        # bd = affinity_to_boundary(masks[surround_coords], affinity[:,*surround_coords], None, dim=dim)
        bd = affinity_to_boundary(masks[surround_coords], affinity[(Ellipsis,)+surround_coords], None, dim=dim)
        
        
        bd = bd*masks[surround_coords]
        
        # print('yoyo', masks.ndim, dim,  self.parent.cellpix.ndim)
        bounds[surround_coords] = bd
        
        
        # also take care of any orphaned masks
        # print(affinity[:,*surround_coords].sum(axis=0).shape, masks[surround_coords].shape)
        
        # masks[*surround_coords][affinity[:,*surround_coords].sum(axis=0)==0] = 0
        # masks[*surround_coords] *= affinity[:,*surround_coords].sum(axis=0)>0
        # masks[*surround_coords] *= affinity[:,*surround_coords].sum(axis=0)>0
        masks[surround_coords] *= affinity[(Ellipsis,)+surround_coords].sum(axis=0)>0
        
    

        # print('info', self.parent.pixelGridOverlay.lineIndices.shape)
        # Update only the affected region of the overlay
        # self.parent.draw_layer(region=(c0, c1, r0, r1), z=z)
        # pass a dilated region to catch the outer edges of the new boundary 
        self.parent.draw_layer(region=(dil_c0, dil_c1, dil_r0, dil_r1), z=z)
    
    
    def _cleanup(self):
        # brush_size = getattr(self.parent, 'brush_size', 1)
        # label = getattr(self.parent, 'current_label', 0)
        z = self.parent.currentZ
        masks = self.parent.cellpix[z]
        bounds = self.parent.outpix[z] # this might be a direct hook to the display
        
        affinity = self.parent.affinity_graph
        height, width = masks.shape
        
        steps = self.parent.steps
        dim = self.parent.dim
        idx = self.parent.inds[0][0]
  
        D = dim
        S = len(steps)
        cutoff = 3**(dim-1) + 1
        
        source_slices, target_slices = [[[[] for _ in range(D)] for _ in range(S)] for _ in range(2)]


        s1,s2,s3 = slice(1,None), slice(0,-1), slice(None,None) # this needs to be generalized to D dimensions
        for i in range(S):
            for j in range(D):
                s = steps[i][j]
                target_slices[i][j], source_slices[i][j] = (s1,s2) if s>0 else (s2,s1) if s<0 else (s3,s3)
                
                    
    
        csum = np.sum(affinity,axis=0)
        keep = csum>=cutoff


        for i in self.parent.non_self:

            step = steps[i]
            # target_coords = tuple(c-s for c,s in zip(source_coords, step))
            # source_coords = tuple(c+s for c,s in zip(source_coords, step))
            
            
            target_slc = (Ellipsis,)+tuple(target_slices[i])
            source_slc = (Ellipsis,)+tuple(source_slices[i])
    
            tuples = self.parent.supporting_inds[i]
            support = np.zeros_like(masks[source_slc],dtype=float)
            n_tuples = len(tuples)
            for j in range(n_tuples):
                f_inds = tuples[j]
                b_inds = tuple(S-1-np.array(tuples[-(j+1)]))
                for f,b in zip(f_inds,b_inds):
                    support += np.logical_and(affinity[f][source_slc], affinity[b][target_slc])
                    

            affinity[i][source_slc] = affinity[-(i+1)][target_slc] = np.logical_and.reduce([affinity[i][source_slc],
                                                                                            affinity[-(i+1)][target_slc], 
                                                                                            masks[source_slc]>0,
                                                                                            masks[target_slc]>0,
                                                                                            keep[source_slc],
                                                                                            keep[target_slc],
                                                                                            support>2
                                                                                            ])
            
        update_inds = []
        update_alpha = []
        
        source_coords = self.parent.coords
        print(source_coords[0].shape, self.parent.indexes.shape, self.parent.neigh_inds.shape)
        
        Ly,Lx = masks.shape
        # have to wait to update affinity after looping over all directions   
        for i in self.parent.non_self[:idx]:
            step = steps[i]
            target_coords = tuple(np.clip(c+s,0,l-1) for c,s,l in zip(source_coords, step,[Ly,Lx]))
            # target_slc = (Ellipsis,)+tuple(target_slices[i])
            # source_slc = (Ellipsis,)+tuple(source_slices[i])
            
            
            # inds = self.parent.pixelGridOverlay.lineIndices[*target_coords,i].tolist()
            inds = self.parent.pixelGridOverlay.lineIndices[target_coords+(i,)].tolist()
            
            opp_target_coords = tuple(np.clip(c-s,0,l-1) for c,s,l in zip(source_coords, step,[Ly,Lx]))
            # inds += self.parent.pixelGridOverlay.lineIndices[*opp_target_coords,i].tolist()
            inds += self.parent.pixelGridOverlay.lineIndices[opp_target_coords+(i,)].tolist()

            update_inds += inds
            update_alpha.append(affinity[i][source_coords])
            update_alpha.append(affinity[i][opp_target_coords])
                    
        self.parent.pixelGridOverlay.hide_lines_batch(update_inds, 
                                                      np.concatenate(update_alpha), 
                                                      visible=False)
    
                
        # I think I am missing the background pixels connected to it
        # maybe the cleanup will do it 

        # I could add a cleanup step here from get_affinity_torch
        # it could operate on just the dilated region to update the affinity graph
                 
    

        
        # some strangeness with outlines on and masks off, as if the masks are zero?
                    
        # update boundary

        # print('\n\nA\n\n', affinity.shape, masks.shape,  masks[surround_coords].shape, affinity[:,*surround_coords].shape)
        # print('surround_coords', surround_coords,targets)
        bd = affinity_to_boundary(masks, affinity, None, dim=dim)
        
        bd = bd*masks
        
        # print('yoyo', masks.ndim, dim,  self.parent.cellpix.ndim)
        bounds = bd
        
        
        # also take care of any orphaned masks
        # print(affinity[:,*surround_coords].sum(axis=0).shape, masks[surround_coords].shape)
        
        # masks[*surround_coords][affinity[:,*surround_coords].sum(axis=0)==0] = 0
        masks *= affinity.sum(axis=0)>0
        

        # Update only the affected region of the overlay
        # self.parent.draw_layer(region=(c0, c1, r0, r1), z=z)
        # pass a dilated region to catch the outer edges of the new boundary 
        self.parent.draw_layer(z=z)


        
    def _floodFill(self, x, y, new_label):        
        """Perform flood fill starting at (x, y) with the given new_label."""
        z = self.parent.currentZ
        array = self.parent.cellpix[z]
        old_label = array[int(y), int(x)]
        
        if old_label == new_label:
            return  # Nothing to change

        # Use an efficient queue-based flood fill
        stack = [(int(y), int(x))]
        while stack:
            cy, cx = stack.pop()
            if 0 <= cy < array.shape[0] and 0 <= cx < array.shape[1] and array[cy, cx] == old_label:
                array[cy, cx] = new_label
                # Add neighbors to the stack
                stack.extend([(cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)])

        self.parent.update_layer()  # Refresh display
        
    def _pickLabel(self, x, y):
        """Set the current label to the value under the cursor."""
        z = self.parent.currentZ
        self.parent.current_label = self.parent.cellpix[z, int(y), int(x)]
        self.parent.update_active_label_field()  # Ensure the input field is updated

    def _generateKernel(self, brush_diameter):
        """
        Generates a circular kernel for the given brush diameter.
        Ensures the diameter is the nearest odd number.
        """
        # Ensure the diameter is an odd number
        diameter = int(round(brush_diameter))  # Round to the nearest integer
        if diameter % 2 == 0:  # Make it odd if it's even
            diameter += 1

        radius = diameter // 2
        y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        self._kernel = (x**2 + y**2 <= radius**2)
        
class RangeSlider(QSlider):
    """ A slider for ranges.

        This class provides a dual-slider for ranges, where there is a defined
        maximum and minimum, as is a normal slider, but instead of having a
        single slider value, there are 2 slider values.

        This class emits the same signals as the QSlider base class, with the
        exception of valueChanged

        Found this slider here: https://www.mail-archive.com/pyqt@riverbankcomputing.com/msg22889.html
        and modified it
    """
    def __init__(self, parent=None, *args):
        super(RangeSlider, self).__init__(*args)

        self._low = self.minimum()
        self._high = self.maximum()

        self.pressed_control = QStyle.SC_None
        self.hover_control = QStyle.SC_None
        self.click_offset = 0

        self.setOrientation(QtCore.Qt.Horizontal)
        self.setTickPosition(QSlider.TicksRight)
        self.setStyleSheet(\
                "QSlider::handle:horizontal {\
                background-color: white;\
                border: 1px solid white;\
                border-radius: 2px;\
                border-color: white;\
                height: 8px;\
                width: 6px;\
                margin: 0px 2; \
                }")


        #self.opt = QStyleOptionSlider()
        #self.opt.orientation=QtCore.Qt.Vertical
        #self.initStyleOption(self.opt)
        # 0 for the low, 1 for the high, -1 for both
        self.active_slider = 0
        self.parent = parent

    def level_change(self):
        if self.parent is not None:
            if self.parent.loaded:
                self.parent.ops_plot = {'saturation': [self._low, self._high]}
                self.parent.saturation[self.parent.currentZ] = [self._low, self._high]
                self.parent.update_plot()

    def low(self):
        return self._low

    def setLow(self, low):
        self._low = low
        self.update()

    def high(self):
        return self._high

    def setHigh(self, high):
        self._high = high
        self.update()

    def paintEvent(self, event):
        # based on http://qt.gitorious.org/qt/qt/blobs/master/src/gui/widgets/qslider.cpp
        painter = QPainter(self)
        style = QApplication.style()

        for i, value in enumerate([self._low, self._high]):
            opt = QStyleOptionSlider()
            self.initStyleOption(opt)

            # Only draw the groove for the first slider so it doesn't get drawn
            # on top of the existing ones every time
            if i == 0:
                opt.subControls = QStyle.SC_SliderHandle#QStyle.SC_SliderGroove | QStyle.SC_SliderHandle
            else:
                opt.subControls = QStyle.SC_SliderHandle

            if self.tickPosition() != self.NoTicks:
                opt.subControls |= QStyle.SC_SliderTickmarks

            if self.pressed_control:
                opt.activeSubControls = self.pressed_control
                opt.state |= QStyle.State_Sunken
            else:
                opt.activeSubControls = self.hover_control

            opt.sliderPosition = int(value)
            opt.sliderValue = int(value)
            style.drawComplexControl(QStyle.CC_Slider, opt, painter, self)


    def mousePressEvent(self, event):
        event.accept()

        style = QApplication.style()
        button = event.button()
        # In a normal slider control, when the user clicks on a point in the
        # slider's total range, but not on the slider part of the control the
        # control would jump the slider value to where the user clicked.
        # For this control, clicks which are not direct hits will slide both
        # slider parts
        if button:
            opt = QStyleOptionSlider()
            self.initStyleOption(opt)

            self.active_slider = -1

            for i, value in enumerate([self._low, self._high]):
                opt.sliderPosition = value
                hit = style.hitTestComplexControl(style.CC_Slider, opt, event.pos(), self)
                if hit == style.SC_SliderHandle:
                    self.active_slider = i
                    self.pressed_control = hit

                    self.triggerAction(self.SliderMove)
                    self.setRepeatAction(self.SliderNoAction)
                    self.setSliderDown(True)

                    break

            if self.active_slider < 0:
                self.pressed_control = QStyle.SC_SliderHandle
                self.click_offset = self.__pixelPosToRangeValue(self.__pick(event.pos()))
                self.triggerAction(self.SliderMove)
                self.setRepeatAction(self.SliderNoAction)
        else:
            event.ignore()

    def mouseMoveEvent(self, event):
        if self.pressed_control != QStyle.SC_SliderHandle:
            event.ignore()
            return

        event.accept()
        new_pos = self.__pixelPosToRangeValue(self.__pick(event.pos()))
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)

        if self.active_slider < 0:
            offset = new_pos - self.click_offset
            self._high += offset
            self._low += offset
            if self._low < self.minimum():
                diff = self.minimum() - self._low
                self._low += diff
                self._high += diff
            if self._high > self.maximum():
                diff = self.maximum() - self._high
                self._low += diff
                self._high += diff
        elif self.active_slider == 0:
            if new_pos >= self._high:
                new_pos = self._high - 1
            self._low = new_pos
        else:
            if new_pos <= self._low:
                new_pos = self._low + 1
            self._high = new_pos

        self.click_offset = new_pos
        self.update()

    def mouseReleaseEvent(self, event):
        self.level_change()

    def __pick(self, pt):
        if self.orientation() == QtCore.Qt.Horizontal:
            return pt.x()
        else:
            return pt.y()


    def __pixelPosToRangeValue(self, pos):
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        style = QApplication.style()

        gr = style.subControlRect(style.CC_Slider, opt, style.SC_SliderGroove, self)
        sr = style.subControlRect(style.CC_Slider, opt, style.SC_SliderHandle, self)

        if self.orientation() == QtCore.Qt.Horizontal:
            slider_length = sr.width()
            slider_min = gr.x()
            slider_max = gr.right() - slider_length + 1
        else:
            slider_length = sr.height()
            slider_min = gr.y()
            slider_max = gr.bottom() - slider_length + 1

        return style.sliderValueFromPosition(self.minimum(), self.maximum(),
                                             pos-slider_min, slider_max-slider_min,
                                             opt.upsideDown)



    
class HistLUT(pg.HistogramLUTItem):
    sigLookupTableChanged = QtCore.pyqtSignal(object)
    sigLevelsChanged = QtCore.pyqtSignal(object)
    sigLevelChangeFinished = QtCore.pyqtSignal(object)

    def __init__(self, image=None, fillHistogram=True, levelMode='mono',
                 gradientPosition='right', orientation='vertical'):
        super().__init__(image=image,fillHistogram=fillHistogram,levelMode=levelMode,
                         gradientPosition=gradientPosition,orientation=orientation)
        
        # self.gradient = GradientEditorItem(orientation=self.gradientPosition)
        # self.gradient = GradEditor(orientation=self.gradientPosition) #overwrite with mine
        
        # Cache the original mouse event methods from the base class.
        self._orig_mousePressEvent = pg.HistogramLUTItem.mousePressEvent
        self._orig_mouseMoveEvent = pg.HistogramLUTItem.mouseMoveEvent
        self._orig_mouseReleaseEvent = pg.HistogramLUTItem.mouseReleaseEvent
        # You can also introduce a flag:
        self.show_histogram = True

    def mousePressEvent(self, event):
        if not self.show_histogram:
            event.accept()  # Block interaction
        else:
            # Call the cached base-class handler explicitly.
            self._orig_mousePressEvent(self, event)

    def mouseMoveEvent(self, event):
        if not self.show_histogram:
            event.accept()
        else:
            self._orig_mouseMoveEvent(self, event)

    def mouseReleaseEvent(self, event):
        if not self.show_histogram:
            event.accept()
        else:
            self._orig_mouseReleaseEvent(self, event)

    def paint(self, p, *args):
        # paint the bounding edges of the region item and gradient item with lines
        # connecting them
        if self.levelMode != 'mono' or not self.region.isVisible():
            return

        pen = self.region.lines[0].pen

        mn, mx = self.getLevels()
        vbc = self.vb.viewRect().center()
        gradRect = self.gradient.mapRectToParent(self.gradient.gradRect.rect())
        if self.orientation == 'vertical':
            p1mn = self.vb.mapFromViewToItem(self, Point(vbc.x(), mn)) + Point(0, 5)
            p1mx = self.vb.mapFromViewToItem(self, Point(vbc.x(), mx)) - Point(0, 5)
            if self.gradientPosition == 'right':
                p2mn = gradRect.bottomLeft()
                p2mx = gradRect.topLeft()
            else:
                p2mn = gradRect.bottomRight()
                p2mx = gradRect.topRight()
        else:
            p1mn = self.vb.mapFromViewToItem(self, Point(mn, vbc.y())) - Point(5, 0)
            p1mx = self.vb.mapFromViewToItem(self, Point(mx, vbc.y())) + Point(5, 0)
            if self.gradientPosition == 'bottom':
                p2mn = gradRect.topLeft()
                p2mx = gradRect.topRight()
            else:
                p2mn = gradRect.bottomLeft()
                p2mx = gradRect.bottomRight()

        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        for pen in [pen]: #get rid of first entry, shadow of some sort
            p.setPen(pen)

            # lines from the linear region item bounds to the gradient item bounds
            p.drawLine(p1mn, p2mn)
            p.drawLine(p1mx, p2mx)

            # lines bounding the edges of the gradient item
            if self.orientation == 'vertical':
                p.drawLine(gradRect.topLeft(), gradRect.topRight())
                p.drawLine(gradRect.bottomLeft(), gradRect.bottomRight())
            else:
                p.drawLine(gradRect.topLeft(), gradRect.bottomLeft())
                p.drawLine(gradRect.topRight(), gradRect.bottomRight())
                
                
        #change where gradienteditoritem is called and pass in my custom one
        
                
class GradEditor(pg.GradientEditorItem):
    sigGradientChanged = QtCore.pyqtSignal(object)
    sigGradientChangeFinished = QtCore.pyqtSignal(object)
    
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
              
                
# class Tick2(pg.Tick):  ## NOTE: Making this a subclass of GraphicsObject instead results in
#                                     ## activating this bug: https://bugreports.qt-project.org/browse/PYSIDE-86
#     ## private class

#     # When making Tick a subclass of QtWidgets.QGraphicsObject as origin,
#     # ..GraphicsScene.items(self, *args) will get Tick object as a
#     # class of QtGui.QMultimediaWidgets.QGraphicsVideoItem in python2.7-PyQt5(5.4.0)

#     sigMoving = QtCore.pyqtSignal(object, object)
#     sigMoved = QtCore.pyqtSignal(object)
#     sigClicked = QtCore.pyqtSignal(object, object)
    
#     # def __init__(self, pos, color, movable=True, scale=10, pen='w', removeAllowed=True):
#     #     super().__init__(pos=pos,color=color,movable=movable,scale=scale,pen=pen,removeAllowed=removeAllowed)
#     def __init__(self,*args):
#         super.__init__(*args)

#     def paint(self, p, *args):
#         p.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing)
#         p.fillPath(self.pg, fn.mkBrush(self.color))
        
#         p.setPen(self.currentPen)
#         p.setHoverPen(self.hoverPen)
#         p.drawPath(self.pg)

from PyQt6.QtWidgets import QPushButton
from PyQt6.QtGui import QPalette
import qtawesome as qta

class IconToggleButton(QPushButton):
    def __init__(self, icon_active_name, icon_inactive_name, active_color=None, 
                 inactive_color="gray", icon_size=20, parent=None):
        super().__init__(parent)
        self.icon_active_name = icon_active_name
        self.icon_inactive_name = icon_inactive_name
        self.active_color = active_color  # Custom active color
        self.inactive_color = inactive_color
        self.icon_size = icon_size

        self.setCheckable(True)
        self.update_icons()  # Set the initial icons

        # Style for left-aligning the icon
        self.setStyleSheet("""
            QPushButton {
                border: none;
                background: none;
                text-align: left;  /* Align text/icon to the left */
                padding-left: 0px; /* Remove any left padding */
            }
        """)

        self.toggled.connect(self.update_icons)

    def update_icons(self):
        """Update the icons dynamically based on the palette and state."""
        # Use the provided active color or fallback to the highlight color
        active_color = (
            self.active_color or self.palette().brush(QPalette.ColorRole.Highlight).color().name()
        )
        
        # Generate icons dynamically
        icon_active = qta.icon(self.icon_active_name, color=active_color)
        icon_inactive = qta.icon(self.icon_inactive_name, color=self.inactive_color)

        # Set the appropriate icon
        self.setIcon(icon_active if self.isChecked() else icon_inactive)
        self.setIconSize(QtCore.QSize(self.icon_size, self.icon_size))

    def _emit_toggled(self):
        """Emit the toggled signal when the button is clicked."""
        self.toggled.emit(self.isChecked())
        
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, QLineF
from PyQt6.QtGui import QPainter, QPen
from pyqtgraph import GraphicsObject
from PyQt6.QtWidgets import QGraphicsItem


from PyQt6.QtCore import Qt
class AffinityOverlay_old(GraphicsObject):
    def __init__(self, pixel_size=1, threshold=0.5, parent=None):
        super().__init__()
        self.parent = parent
        self.pixel_size = pixel_size
        self.threshold = threshold
        self.pen = QPen()
        # self.pen.setColor(Qt.GlobalColor.white)
        # self.pen.setWidth(.1)
        self._lines = None
        # self._generate_lines()
        # Turn the overlay off by default:
        self.setVisible(False)
        
        # try to avoid redraw
        self.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)
        # self.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)

    # def _generate_lines(self):
    #     offsets = self.parent.steps
    #     print('generate lines', offsets, self.pixel_size)
    #     d, Y, X = self.parent.affinity_graph.shape
    #     xs = np.arange(X) * self.pixel_size + self.pixel_size / 2
    #     ys = np.arange(Y) * self.pixel_size + self.pixel_size / 2
    #     grid_x, grid_y = np.meshgrid(xs, ys)
    #     lines = []
        
    #     idx = self.parent.inds[0][0]
    #     z = 0
    #     source_mask = self.parent.cellpix[z]>0
        
    #     # for direction in self.parent.non_self:
    #     for direction in self.parent.non_self[:idx]:
    #     # for direction in self.parent.non_self[:2]:
        
        
    #         print('direction', direction)
    #         if direction == idx:
    #             continue
    #         target_mask = self.parent.affinity_graph[direction] 
            
    #         if not np.any(target_mask):
    #             continue
                
    #         mask = np.logical_and(source_mask, target_mask>0)
    #         start_x = grid_x[mask]
    #         start_y = grid_y[mask]
            
    #         dy, dx = offsets[direction] * self.pixel_size
    #         end_x = start_x + dx
    #         end_y = start_y + dy
            
            
    #         for sx, sy, ex, ey in zip(start_x, start_y, end_x, end_y):
    #             lines.append(QLineF(QPointF(sx, sy), QPointF(ex, ey)))
    #     self._lines = lines
    
    def _generate_lines(self):
        offsets = self.parent.steps
        print('generate lines', offsets, self.pixel_size)
        d, Y, X = self.parent.affinity_graph.shape
        xs = np.arange(X) * self.pixel_size + self.pixel_size / 2
        ys = np.arange(Y) * self.pixel_size + self.pixel_size / 2
        grid_x, grid_y = np.meshgrid(xs, ys)
        lines = []
        
        idx = self.parent.inds[0][0]
        z = 0
        source_mask = self.parent.cellpix[z]>0
        
        S, Y, X = self.parent.affinity_graph.shape
        pixel_size = self.pixel_size
        self.line_items = {i:None for i in self.parent.non_self}
        print('generate lines', self.parent.steps, self.pixel_size)

        # Build a grid of pixel centers.
        xs = np.arange(X) * pixel_size + pixel_size / 2
        ys = np.arange(Y) * pixel_size + pixel_size / 2
        grid_x, grid_y = np.meshgrid(xs, ys)  # both shape (Y, X)
        grid_x_flat = grid_x.ravel()
        grid_y_flat = grid_y.ravel()

        # Loop over each desired direction (from non_self).
        for direction in self.parent.non_self[:idx]:
            # Get the offset for this direction and scale by pixel size.
            offset = self.parent.steps[direction]  # e.g. (dy, dx)
            dy, dx = offset * pixel_size

            # Compute the endpoints for every pixel.
            x_end = grid_x_flat + dx
            y_end = grid_y_flat + dy

            # Optionally, filter out segments that go out of bounds.
            in_bounds = ((x_end >= 0) & (x_end < X * pixel_size) &
                         (y_end >= 0) & (y_end < Y * pixel_size))
            x_start_valid = grid_x_flat[in_bounds]
            y_start_valid = grid_y_flat[in_bounds]
            x_end_valid   = x_end[in_bounds]
            y_end_valid   = y_end[in_bounds]
            
            
            for sx, sy, ex, ey in zip(x_start_valid, y_start_valid, x_end_valid, y_end_valid):
                lines.append(QLineF(QPointF(sx, sy), QPointF(ex, ey)))
        self._lines = lines

    def shape(self):
        # Return an empty shape if you never need mouse interaction
        path = QtGui.QPainterPath()
        return path

    def boundingRect(self):
        Y, X = self.parent.affinity_graph.shape[1], self.parent.affinity_graph.shape[2]
        return QRectF(0, 0, X * self.pixel_size, Y * self.pixel_size)

    
    def paint(self, painter, option, widget=None):
        painter.setPen(self.pen)
        if self._lines is None:
            self._generate_lines()
        print('drawing lines', len(self._lines))
        painter.drawLines(self._lines)

    def toggle(self, visible=None):
        """Toggle the overlay on and off. If 'visible' is provided, use it; otherwise, toggle current state."""
        print('ss')
        self.pen.setColor(Qt.GlobalColor.white)
        self.pen.setWidthF(.01)         # or 0.5, etc.
        self.pen.setCosmetic(False)      # always 1px on-screen
        self.pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        self.pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        
        if visible is None:
            visible = not self.isVisible()
        self.setVisible(visible)
        # print('aa')
        self._generate_lines()
        self.update()  # schedule a real paint event



   
import numpy as np
from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QPainterPath, QPen, QPolygonF
from PyQt6.QtWidgets import QGraphicsPathItem, QGraphicsView, QGraphicsScene
import pyqtgraph as pg

from PyQt6 import sip

class AffinityOverlay_new(GraphicsObject):
    """
    Holds separate QGraphicsPathItems for each 'direction' (0,1,2,3).
    Toggling each direction simply sets that pathItem's visibility.
    """
    def __init__(self, 
                 pixel_size=1.0,
                 parent=None):
        """
        directions: list of (dx, dy) offsets from each pixel
        """
        super().__init__()

        self.parent = parent
        self.pixel_size = pixel_size
        self.items = []
        self.pen = QPen()
        
        
        self.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)
        

    
    def add_to_scene(self, scene):
        """Add all direction items to the QGraphicsScene."""
        self.scene = scene
        for item in self.items:
            scene.addItem(item)

    def toggle_direction(self, dir_index, visible=None):
        """
        Toggle the path item at directions[dir_index].
        If visible is None, flip the current visibility;
        otherwise set it to the provided bool.
        """
        item = self.items[dir_index]
        if visible is None:
            visible = not item.isVisible()
        item.setVisible(visible)
    
    def toggle(self, on=True):
        """Convenience to show/hide all directions at once."""
        print('ss', len(self.items))

        if len(self.items)==0:
            self._generate_lines()
        
        for item in self.items:
            item.setVisible(on)
        print(self.parent.indexes.shape, self.parent.neigh_inds.shape, self.parent.ind_matrix.shape, self.parent.coords[0].shape)
        self.update()  # schedule a real paint event

    
    def build_line_arrays(self, x_start, y_start, x_end, y_end):
        """
        Given arrays of start and end coordinates for a set of line segments,
        build interleaved arrays that PlotCurveItem can draw using connect='pairs'.
        """
        N = len(x_start)
        xx = np.empty(2 * N, dtype=float)
        yy = np.empty(2 * N, dtype=float)
        xx[0::2] = x_start
        yy[0::2] = y_start
        xx[1::2] = x_end
        yy[1::2] = y_end
        return xx, yy

    def generate_affinity_lines(self):
        """
        Compute the line segments for each direction (except self-connection)
        in a fully vectorized way and create a PlotCurveItem for each.
        The items are stored in self.line_items and added to the parent's view.
        """
        # Get image dimensions from the affinity graph shape.
        # affinity_graph is assumed to have shape (S, Y, X)
        S, Y, X = self.parent.affinity_graph.shape
        pixel_size = self.pixel_size
        self.line_items = {i:None for i in self.parent.non_self}
        print('generate lines', self.parent.steps, self.pixel_size)

        # Build a grid of pixel centers.
        xs = np.arange(X) * pixel_size + pixel_size / 2
        ys = np.arange(Y) * pixel_size + pixel_size / 2
        grid_x, grid_y = np.meshgrid(xs, ys)  # both shape (Y, X)
        grid_x_flat = grid_x.ravel()
        grid_y_flat = grid_y.ravel()

        # Loop over each desired direction (from non_self).
        for direction in self.parent.non_self:
            # Get the offset for this direction and scale by pixel size.
            offset = self.parent.steps[direction]  # e.g. (dy, dx)
            dy, dx = offset * pixel_size

            # Compute the endpoints for every pixel.
            x_end = grid_x_flat + dx
            y_end = grid_y_flat + dy

            # Optionally, filter out segments that go out of bounds.
            in_bounds = ((x_end >= 0) & (x_end < X * pixel_size) &
                         (y_end >= 0) & (y_end < Y * pixel_size))
            x_start_valid = grid_x_flat[in_bounds]
            y_start_valid = grid_y_flat[in_bounds]
            x_end_valid   = x_end[in_bounds]
            y_end_valid   = y_end[in_bounds]

            # Build interleaved coordinate arrays.
            xx, yy = self.build_line_arrays(x_start_valid, y_start_valid,
                                            x_end_valid, y_end_valid)

            # Create a PlotCurveItem using connect='pairs'.
            pen = pg.mkPen('red', width=1.0,
                           cap=QtCore.Qt.PenCapStyle.RoundCap,
                           join=QtCore.Qt.PenJoinStyle.RoundJoin)
            item = pg.PlotCurveItem(xx, yy, connect='pairs', pen=pen)
            item.setVisible(False)  # start hidden
            item.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)

            # Store the item in the dictionary keyed by direction.
            self.line_items[direction] = item

            # Add the item to the parent's graphics view (assumed to be self.parent.p0).
            self.parent.p0.addItem(item)
    
    # def array_to_qpolygonf(self,pts: np.ndarray) -> QPolygonF:
    #     """
    #     Convert a NumPy array of shape (npts, 2) into a QPolygonF.
    #     This version explicitly specifies the number of bytes so that the
    #     sip.voidptr has a known size.
    #     """
    #     poly = QPolygonF()
    #     npts = pts.shape[0]
    #     poly.resize(npts)
    #     total_bytes = npts * 2 * 8  # npts * 2 coordinates * 8 bytes per double
    #     vp = sip.voidptr(poly.data(), total_bytes, True)
    #     mem = np.frombuffer(vp, dtype=np.float64).reshape((npts, 2))
    #     mem[:] = pts
    #     return poly
        
        
    # def build_line_arrays(self, x_start, y_start, x_end, y_end):
    #     """
    #     Given 1D arrays x_start, y_start, x_end, y_end (all same length),
    #     build an interleaved (2*N,2) array where each consecutive pair defines a line.
    #     """
    #     N = x_start.size
    #     pts = np.empty((2 * N, 2), dtype=np.float64)
    #     pts[0::2, 0] = x_start
    #     pts[0::2, 1] = y_start
    #     pts[1::2, 0] = x_end
    #     pts[1::2, 1] = y_end
    #     # return pts
                
    #     # """
    #     # Given four 1D NumPy arrays (start_x, start_y, end_x, end_y) of equal length,
    #     # interleave them into a (2*N, 2) array where each consecutive pair defines a line.
    #     # Returns a QPolygonF built from this array.
    #     # """
    #     # N = start_x.size
    #     # pts = np.empty((2 * N, 2), dtype=np.float64)
    #     # pts[0::2, 0] = start_x
    #     # pts[0::2, 1] = start_y
    #     # pts[1::2, 0] = end_x
    #     # pts[1::2, 1] = end_y
    #     return self.array_to_qpolygonf(pts)


    # def generate_affinity_lines(self):
    #     """
    #     Compute the line segments for each direction (from self.parent.non_self)
    #     in a fully vectorized way and create a PlotCurveItem for each.
    #     The items are stored in self.line_items and added to the parent's view.
    #     """
    #     # Affinity graph is assumed to have shape (S, Y, X)
    #     S, Y, X = self.parent.affinity_graph.shape
    #     pixel_size = self.pixel_size

    #     # Build a grid of pixel centers.
    #     xs = np.arange(X) * pixel_size + pixel_size / 2
    #     ys = np.arange(Y) * pixel_size + pixel_size / 2
    #     grid_x, grid_y = np.meshgrid(xs, ys)  # shape (Y, X)
    #     grid_x_flat = grid_x.ravel()
    #     grid_y_flat = grid_y.ravel()

    #     # Initialize the dictionary to hold items.
    #     self.line_items = {}

    #     # Loop over each desired direction.
    #     for direction in self.parent.non_self:
    #         # Get the offset for this direction and scale by pixel size.
    #         offset = self.parent.steps[direction]  # e.g. (dy, dx)
    #         dy, dx = offset * pixel_size

    #         # Compute endpoints for every pixel.
    #         x_end = grid_x_flat + dx
    #         y_end = grid_y_flat + dy

    #         # Filter out segments that go out of bounds.
    #         in_bounds = ((x_end >= 0) & (x_end < X * pixel_size) &
    #                      (y_end >= 0) & (y_end < Y * pixel_size))
    #         x_start_valid = grid_x_flat[in_bounds]
    #         y_start_valid = grid_y_flat[in_bounds]
    #         x_end_valid   = x_end[in_bounds]
    #         y_end_valid   = y_end[in_bounds]

    #         # Build interleaved coordinate array.
    #         # pts = self.build_line_arrays(x_start_valid, y_start_valid,
    #         #                              x_end_valid, y_end_valid)
    #         # # Convert to QPolygonF in a vectorized manner.
    #         # poly = self.array_to_qpolygonf(pts)
            
    #         # Build interleaved coordinate array (already returns a QPolygonF).
    #         poly = self.build_line_arrays(x_start_valid, y_start_valid,
    #                                     x_end_valid, y_end_valid)

    #         # Create a PlotCurveItem using connect='pairs'
    #         pen = pg.mkPen('red', width=1.0,
    #                        cap=QtCore.Qt.PenCapStyle.RoundCap,
    #                        join=QtCore.Qt.PenJoinStyle.RoundJoin)
    #         item = pg.PlotCurveItem(poly, connect='pairs', pen=pen)
    #         item.setVisible(False)
    #         item.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)

    #         # Store the item for this direction.
    #         self.line_items[direction] = item
    #         # Add the item to the parent's graphics view (assumed to be self.parent.p0).
    #         self.parent.p0.addItem(item)

    # def toggle(self, visible=True):
    #     """
    #     Toggle the visibility of all affinity line items.
    #     Call this after generate_affinity_lines to show or hide them.
    #     """
    #     for item in self.line_items.values():
    #         item.setVisible(visible)
    #     self.update()

    def toggle(self,visible=True):
        """
        (Optional) Update the appearance or data of the line items.
        For example, you could recolor lines based on a threshold from the affinity graph.
        This function demonstrates how you might loop over each item and update its pen.
        """
        # if not hasattr(self, 'line_items'):
        self.generate_affinity_lines()
        
        self.pen.setColor(Qt.GlobalColor.white)
        # self.pen.setWidth(.1)
        self.pen.setWidthF(1.0)         # or 0.5, etc.
        self.pen.setCosmetic(True)      # always 1px on-screen
        self.pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        self.pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        
        print('ss', visible)
        for direction, item in self.line_items.items():
            # Here you can implement logic to update item data or appearance.
            # For instance, change the pen color if needed.
            
            
            # pen = pg.mkPen('red', width=1.0,
            #                cap=QtCore.Qt.PenCapStyle.RoundCap,
            #                join=QtCore.Qt.PenJoinStyle.RoundJoin)
            item.setPen(self.pen)
            item.setVisible(visible)  # start hidden

        self.update()
    
    
    def _generate_lines(self):
    
        print('generating lines')
        self.steps = self.parent.steps # can truncate to first 4

        # Y, X = self.parent.affinity_graph.shape[1], self.parent.affinity_graph.shape[2]
        d, Y, X = self.parent.affinity_graph.shape

        self.pen = QPen()
        self.pen.setColor(pg.mkColor('white'))
        self.pen.setWidthF(1.0)
        self.pen.setCosmetic(True)  # always 1-pixel thick regardless of zoom
        pixel_size = self.pixel_size
        # Pre-build each direction’s path
        
        source = self.parent.coords
        neigh_inds = self.parent.neigh_inds
        self.lines = []
        for neigh in neigh_inds:
            print('a')
            target = tuple(c[neigh] for c in coords)
            
        
        
        for i, (dy, dx) in enumerate(self.steps):
            path_item = QGraphicsPathItem()
            path_item.setPen(self.pen)
            path_item.setVisible(False)  # start off hidden
            path = QPainterPath()

            # For each pixel (x,y), create a short line from (x,y) to (x+dx, y+dy)
            # (Here we do a small subset example; in real code, do all valid pixels.)
            for y in range(Y):
                for x in range(X):
                    # Starting point is center of the pixel
                    sx = x * pixel_size + pixel_size/2
                    sy = y * pixel_size + pixel_size/2
                    ex = sx + dx * pixel_size
                    ey = sy + dy * pixel_size

                    # Move, then draw line
                    path.moveTo(sx, sy)
                    path.lineTo(ex, ey)
            
            path_item.setPath(path)
            self.items.append(path_item)
            
        # print('len items', len(self.items))
        

        # offsets = self.parent.steps
        # print('generate lines', offsets, self.pixel_size)
        # xs = np.arange(X) * self.pixel_size + self.pixel_size / 2
        # ys = np.arange(Y) * self.pixel_size + self.pixel_size / 2
        # grid_x, grid_y = np.meshgrid(xs, ys)
        # lines = []
        
        # idx = self.parent.inds[0][0]
        # z = 0
        # source_mask = self.parent.cellpix[z]>0

        
        # # for direction in self.parent.non_self:
        # for direction in self.parent.non_self[:idx]:
        # # for direction in self.parent.non_self[:2]:
        
        
        #     print('direction', direction)
        #     if direction == idx:
        #         continue
        #     target_mask = self.parent.affinity_graph[direction] 
            
        #     if not np.any(target_mask):
        #         continue
                
        #     mask = np.logical_and(source_mask, target_mask>0)
        #     start_x = grid_x[mask]
        #     start_y = grid_y[mask]
            
        #     dy, dx = offsets[direction] * self.pixel_size
        #     end_x = start_x + dx
        #     end_y = start_y + dy
            
            
        #     for sx, sy, ex, ey in zip(start_x, start_y, end_x, end_y):
        #         lines.append(QLineF(QPointF(sx, sy), QPointF(ex, ey)))
        # self._lines = lines
        

import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np

class AffinityOverlay_GL:
    """
    An OpenGL-based overlay that draws millions of short line segments efficiently.
    """
    def __init__(self, parent=None, pixel_size=1.0):
        # 'parent' could be your main GUI or data holder
        self.parent = parent
        self.pixel_size = pixel_size

        # Create a GLViewWidget to host the lines
        self.gl_view = gl.GLViewWidget()

        # Optional: configure camera or disable rotation if purely 2D
        # self.gl_view.orbit(0, 0)   # choose orientation
        # self.gl_view.setCameraPosition(distance=some_value)

        # You can keep a list/dict of GLLinePlotItem
        self.line_item = None

        # By default, hide the entire GL widget
        self.gl_view.setVisible(False)
        
        self.zValue = 0

    def generate_segments(self):
        """
        Example that creates Nx2 or Nx3 array of vertex pairs, shaped (2*N,3)
        for 'connect="pairs"'. We treat this as 3D coords in XY plane at Z=0.
        """
        # Suppose your parent has shape info:
        S, Y, X = self.parent.affinity_graph.shape  # Example shape: (num_directions, Y, X)
        steps = self.parent.steps                   # your offset vectors: e.g. (dy, dx)
        pixel_size = self.pixel_size

        # Make a grid of center points (x, y)
        xs = np.arange(X) * pixel_size + (pixel_size/2)
        ys = np.arange(Y) * pixel_size + (pixel_size/2)
        grid_x, grid_y = np.meshgrid(xs, ys)
        grid_x = grid_x.ravel()
        grid_y = grid_y.ravel()

        # For demonstration, just build a single huge array of line endpoints
        # from multiple directions. For extremely large data, you might chunk or
        # compress. We'll do 'pairs' so that every two consecutive points forms
        # a line segment.

        big_list = []
        for direction in self.parent.non_self:
            dy, dx = steps[direction] * pixel_size
            x_end = grid_x + dx
            y_end = grid_y + dy

            # Filter out-of-bounds if desired
            in_bounds = ((x_end >= 0) & (x_end < X*pixel_size) &
                         (y_end >= 0) & (y_end < Y*pixel_size))
            x_s = grid_x[in_bounds]
            y_s = grid_y[in_bounds]
            x_e = x_end[in_bounds]
            y_e = y_end[in_bounds]

            # Interleave [start, end, start, end, ...] in Nx3
            # Z=0 for all points
            seg_count = len(x_s)
            coords = np.zeros((2*seg_count, 3), dtype=np.float32)
            coords[0::2, 0] = x_s  # x-start
            coords[0::2, 1] = y_s  # y-start
            coords[1::2, 0] = x_e  # x-end
            coords[1::2, 1] = y_e  # y-end
            big_list.append(coords)

        # Concatenate everything
        all_coords = np.concatenate(big_list, axis=0)
        return all_coords

    def create_line_item(self):
        """
        Create or update the GLLinePlotItem from the line coordinates array.
        """
        coords = self.generate_segments()
        # We'll color all lines white, for example
        color = (1.0, 1.0, 1.0, 1.0)  # RGBA

        # If you want 1-pixel width lines, set width=1.0 or so
        self.line_item = gl.GLLinePlotItem(
            pos=coords,         # Nx3 array
            color=color,        
            width=1.0,          # in pixel units
            antialias=True,
            mode='lines'        # or 'line_strip'
        )

        # Add to the GL view
        self.gl_view.addItem(self.line_item)

    def toggle(self, visible=None):
        """
        Toggle the overlay on/off. If 'visible' is None, flip current state.
        """
        current = self.gl_view.isVisible()
        if visible is None:
            visible = not current
        self.gl_view.setVisible(visible)

        if visible and self.line_item is None:
            self.create_line_item()  # build the line buffer

    def clear(self):
        """
        Remove lines from the scene if you need to rebuild from scratch.
        """
        if self.line_item is not None:
            self.gl_view.removeItem(self.line_item)
            self.line_item = None

    # Optionally, provide a method for your main GUI to retrieve the widget itself:
    def get_widget(self):
        return self.gl_view

import numpy as np
import pyqtgraph as pg
pg.setConfigOptions(useOpenGL=True)  # Attempt to enable OpenGL in pyqtgraph

from pyqtgraph import GraphicsObject
from PyQt6.QtWidgets import QGraphicsItem
from PyQt6.QtGui import QPainter
from PyQt6.QtCore import QRectF, Qt

import OpenGL.GL as gl

class GLLineOverlay(GraphicsObject):
    """
    A QGraphicsItem in pyqtgraph that uses raw OpenGL calls (via PyOpenGL)
    to draw lines in an overlay. It draws two diagonals from (0,0) to (width,height)
    and (0,height) to (width,0). 
    """

    def __init__(self, width=100, height=100, parent=None):
        super().__init__()
        self.parent = parent
        self.width = width
        self.height = height
        
        # No mouse interaction
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

        # Build the data for two lines (4 points total).
        # Each line has 2 vertices => total 2*N points for N lines.
        # lines: (0,0)->(width,height), (0,height)->(width,0)
        self.vertex_data = np.array([
            [0,          0],
            [self.width, self.height],
            [0,          self.height],
            [self.width, 0],
        ], dtype=np.float32)

        # Create a Vertex Buffer Object (VBO) for the line data
        self.vbo_id = gl.glGenBuffers(1)
        self.upload_data()

    def upload_data(self):
        """Send our vertex_data to the GPU in a VBO."""
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_id)
        gl.glBufferData(gl.GL_ARRAY_BUFFER,
                        self.vertex_data.nbytes,
                        self.vertex_data,
                        gl.GL_STATIC_DRAW)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

    def update_size(self, width, height):
        print('yo')
        """If your underlying image/scene size changes, rebuild the lines."""
        self.width = width
        self.height = height
        # Update line endpoints
        self.vertex_data = np.array([
            [0,          0],
            [self.width, self.height],
            [0,          self.height],
            [self.width, 0],
        ], dtype=np.float32)
        self.upload_data()
        self.update()  # schedule a repaint

    def boundingRect(self):
        """
        Return the bounding rectangle in *local item coordinates*.
        This is used for scene culling and mouse picking (if applicable).
        """
        return QRectF(0, 0, self.width, self.height)

    def paint(self, painter, option, widget=None):
        """
        Called by the GraphicsView to draw this item.
        We wrap raw OpenGL calls with QPainter's native-painting block.
        """
        painter.beginNativePainting()
        try:
            # Enable fixed-function vertex array
            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)

            # Bind our VBO and point the OpenGL client state to it
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_id)
            gl.glVertexPointer(2, gl.GL_FLOAT, 0, None)

            # Basic styling
            gl.glLineWidth(1.0)
            gl.glColor4f(1.0, 1.0, 1.0, 1.0)  # White lines

            # Draw: 4 vertices => 2 lines
            gl.glDrawArrays(gl.GL_LINES, 0, 4)

            # Unbind
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
            gl.glDisableClientState(gl.GL_VERTEX_ARRAY)

        finally:
            painter.endNativePainting()

    def shape(self):
        """
        If we needed mouse picking, we'd return a QPainterPath here.
        We return an empty path because we don't want interaction.
        """
        return pg.QtGui.QPainterPath()

    def setZValue(self, z):
        """Let this overlay sit on top of other items if needed."""
        super().setZValue(z)
        
import numpy as np
import pyqtgraph as pg
from pyqtgraph import GraphicsObject
from PyQt6.QtWidgets import QGraphicsItem
from PyQt6.QtGui import QPainter
from PyQt6.QtCore import QRectF, Qt

import OpenGL.GL as gl

import random
class GLPixelGridOverlay(GraphicsObject):
    """
    A QGraphicsItem that draws lines between centers of adjacent pixels
    (including diagonal neighbors) in an Nx x Ny grid, using raw OpenGL calls.
    """

    def __init__(self, Nx=100, Ny=80, parent=None):
        super().__init__()
        self.parent = parent
        self.Nx = Nx
        self.Ny = Ny

        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

        # Build vertex data (8-neighbor)
        self.vertex_data = self._generate_grid_lines_8()
        
        # Create VBO
        self.vbo_id = gl.glGenBuffers(1)
        self.upload_data()
        
        # Suppose you have self.num_lines = N
        # So self.vertex_data.shape = (2*N, 2)
        # Create color array shape (2*N, 4), all white
        self.num_lines = self.vertex_data.shape[0] // 2
        self.color_data = np.zeros((2*self.num_lines, 4), dtype=np.float32)
        self.base_alpha = np.ones_like(self.color_data[:, 3])  # all 1.0 initially

        # Create a VBO for color
        self.color_id = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.color_id)
        gl.glBufferData(gl.GL_ARRAY_BUFFER,
                        self.color_data.nbytes,
                        self.color_data,
                        gl.GL_DYNAMIC_DRAW)  # dynamic, because we'll update it frequently
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

    def _generate_grid_lines_8(self):
        """
        Build a (2*E, 2) float array of line endpoints for 8-neighbor adjacency,
        plus a 3D array lineIndices of shape (Ny, Nx, ndirs) storing the line index.
        We only go 'forward' directions to avoid duplicates.
        """
        # Suppose 'idx' is how many offsets you want:
        idx = self.parent.inds[0][0]  
        neighbor_offsets = self.parent.steps[:idx]
        print("Using offsets:", neighbor_offsets)
        
        coords = []
        # lineIndices[j, i, d] -> which line index is used for pixel (i,j) and direction d
        ndirs = len(neighbor_offsets)
        lineIndices = np.full((self.Ny, self.Nx, ndirs), -1, dtype=int)

        line_index = 0
        
        for j in range(self.Ny):
            for i in range(self.Nx):
                # center of pixel (i,j)
                cx = i + 0.5
                cy = j + 0.5

                # loop over each direction
                for d, (dy, dx) in enumerate(neighbor_offsets):
                    ni = i + dx
                    nj = j + dy

                    # is neighbor in bounds?
                    if (0 <= ni < self.Nx) and (0 <= nj < self.Ny):
                        nx = ni + 0.5
                        ny = nj + 0.5

                        # We add two consecutive vertices => one line in the final buffer
                        coords.append([cx, cy])
                        coords.append([nx, ny])

                        # record line index in lineIndices array
                        lineIndices[j, i, d] = line_index
                        line_index += 1

        # Convert coords to float32 array
        coords = np.array(coords, dtype=np.float32)
        print("Total lines:", line_index, "vertex_data shape:", coords.shape)

        # Store lineIndices for future lookups
        self.lineIndices = lineIndices
        self.num_lines = line_index

        return coords

    def upload_data(self):
        """Upload the vertex data into a GPU buffer (VBO)."""
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_id)
        gl.glBufferData(gl.GL_ARRAY_BUFFER,
                        self.vertex_data.nbytes,
                        self.vertex_data,
                        gl.GL_STATIC_DRAW)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

    def boundingRect(self):
        """
        The bounding rect in local coords from (0,0) to (Nx, Ny).
        Must match your image’s coordinate system if you want these lines 
        exactly over the image.
        """
        return QRectF(0, 0, self.Nx, self.Ny)

    
    def paint(self, painter, option, widget=None):
        # 1) Temporarily exit native painting so we can do CPU updates (not strictly required,
        #    but often safer if you do QPainter stuff).
        # painter.endNativePainting()

        # 2) Read the transform scale from the painter
        transform = painter.transform()
        scale = transform.m11()  # no rotation/shear assumed

        # print('yo', self.parent.coords[0].shape)
        # self._generate_grid_lines_8()
        # 3) Compute a smooth alpha factor from scale via a sigmoid
        #    For example, fade out below 'threshold=1.0' with 'steepness=4.0'
        alpha_scale = self.scale_to_alpha(scale, threshold=10.0, steepness=0.5)
        # 4) Multiply the original alpha (self.base_alpha) by alpha_scale
        #    so lines that were hidden remain hidden (base_alpha=0).
        final_alpha = self.base_alpha * alpha_scale

        # 5) Update just the alpha channel in self.color_data
        self.color_data[:, 3] = final_alpha

        # 6) Upload the color data to the GPU in one call
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.color_id)
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, self.color_data.nbytes, self.color_data)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

        # 7) Re-enter native painting for raw OpenGL calls
        painter.beginNativePainting()
        try:
            # Enable scissor if you want the viewbox clipping
            gl.glEnable(gl.GL_SCISSOR_TEST)
            gl.glEnable(gl.GL_MULTISAMPLE)

            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

            # Enable vertex array (positions)
            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_id)
            gl.glVertexPointer(2, gl.GL_FLOAT, 0, None)

            # Enable color array (per-vertex RGBA)
            gl.glEnableClientState(gl.GL_COLOR_ARRAY)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.color_id)
            gl.glColorPointer(4, gl.GL_FLOAT, 0, None)

            # Adjust line width to mimic a cosmetic pen
            inverse_scale = 1.0 / scale if scale > 0 else 1.0
            dpix = self.parent.devicePixelRatio()
            line_width = 1.0 * inverse_scale * dpix
            gl.glLineWidth(line_width)

            # Draw all lines
            gl.glDrawArrays(gl.GL_LINES, 0, 2 * self.num_lines)

            # Cleanup
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
            gl.glDisableClientState(gl.GL_COLOR_ARRAY)
            gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        finally:
            gl.glDisable(gl.GL_MULTISAMPLE)
            gl.glDisable(gl.GL_SCISSOR_TEST)
            gl.glDisable(gl.GL_BLEND)
            painter.endNativePainting()  
    
    def hide_lines_batch(self, indices, alpha=None, visible=False):
        """
        Hide or show all the lines in `indices` with a single GPU update.
        If visible=False, set alpha=0 for each line (i.e. line is invisible).
        If visible=True, set alpha=1 for each line (line is fully visible).
        
        Parameters
        ----------
        indices : list or array
            List of line indices to hide/show. Each line i corresponds to
            vertices [2*i, 2*i+1] in color_data.
        visible : bool
            If False, alpha=0 (invisible). If True, alpha=1 (visible).
        """
        new_alpha = 1.0 if visible else 0.0
        
        if alpha is None:
            alpha = [1.0]*len(indices)

        # 1) Modify self.color_data in CPU memory.
        for i,a in zip(indices, alpha):
            # Each line i has two vertices => color_data[2*i : 2*i+2].
            # RGBA => color_data[...] shape is (N,4).
            self.color_data[2*i : 2*i+2, 3] = new_alpha
            
            self.color_data[2*i : 2*i+2] = [a,a,a,a]
        
        # 2) Update GPU buffer. For simplicity, re-upload the entire color array:
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.color_id)
        gl.glBufferSubData(
            gl.GL_ARRAY_BUFFER,
            0,
            self.color_data.nbytes,
            self.color_data
        )
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        # 3) Trigger a redraw just once
        self.update()
        
        
    def show_all_lines(self):
        # show all lines: just do alpha=1 for everything
        self.color_data[:, 3] = 1
        # print(self.color_data.shape)
        # self.color_data[:,1] = 0
        # self.color_data[:,0] = 0
        
        
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.color_id)
        gl.glBufferSubData(
            gl.GL_ARRAY_BUFFER,
            0,
            self.color_data.nbytes,
            self.color_data
        )
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        self.update()
        
    
    def scale_to_alpha(self, scale, threshold=1.0, steepness=4.0):
        """
        Returns a value in [0..1] that smoothly ramps up around 'threshold'.
        alpha = 1 / (1 + exp(-steepness*(scale - threshold)))
        """
        return 1.0 / (1.0 + np.exp(-steepness * (scale - threshold)))
        
    def shape(self):
        """No mouse picking; return empty path."""
        return pg.QtGui.QPainterPath()