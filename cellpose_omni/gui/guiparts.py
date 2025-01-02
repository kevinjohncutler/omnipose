from PyQt6 import QtGui, QtCore
from PyQt6.QtGui import QPainter
from PyQt6.QtWidgets import QApplication, QRadioButton, QWidget, QDialog, QButtonGroup, QSlider, QStyle, QStyleOptionSlider, QGridLayout, QPushButton, QLabel, QLineEdit, QDialogButtonBox, QComboBox
import pyqtgraph as pg
from pyqtgraph import Point
import numpy as np
import os

# import superqt

TOOLBAR_WIDTH = 7
SPACING = 3
WIDTH_0 = 25

from PyQt6.QtWidgets import QPlainTextEdit, QFrame

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
        self.setStyleSheet(parent.styleInactive)
        self.setText(text)
        self.setFont(parent.smallfont)
        self.clicked.connect(lambda: self.press(parent))
        self.model_name = model_name
        
    def press(self, parent):
        for i in range(len(parent.StyleButtons)):
            parent.StyleButtons[i].setStyleSheet(parent.styleUnpressed)
        self.setStyleSheet(parent.stylePressed)
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
            button.setStyleSheet('color: rgb(190,190,190);')
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


class ViewBoxNoRightDrag(pg.ViewBox): # not used anymore 
    def __init__(self, parent=None, border=None, lockAspect=False, enableMouse=True, invertY=False, enableMenu=True, name=None, invertX=False):
        pg.ViewBox.__init__(self, None, border, lockAspect, enableMouse,
                            invertY, enableMenu, name, invertX)
        self.parent = parent
        self.axHistoryPointer = -1

    def keyPressEvent(self, ev):
        """
        This routine should capture key presses in the current view box.
        The following events are implemented:
        +/= : moves forward in the zooming stack (if it exists)
        - : moves backward in the zooming stack (if it exists)

        """
        ev.accept()
        if ev.text() == '-':
            self.scaleBy([1.1, 1.1])
        elif ev.text() in ['+', '=']:
            self.scaleBy([0.9, 0.9])
        else:
            ev.ignore()
    
    def mouseDragEvent(self, ev, axis=None):
        ## if axis is specified, event will only affect that axis.
        if self.parent is None or (self.parent is not None and not self.parent.in_stroke):
            ev.accept()  ## we accept all buttons

            pos = ev.pos()
            lastPos = ev.lastPos()
            dif = pos - lastPos
            dif = dif * -1

            ## Ignore axes if mouse is disabled
            mouseEnabled = np.array(self.state['mouseEnabled'], dtype=np.float)
            mask = mouseEnabled.copy()
            if axis is not None:
                mask[1-axis] = 0.0

            ## Scale or translate based on mouse button
            if ev.button() & (QtCore.Qt.LeftButton | QtCore.Qt.MiddleButton):
                if self.state['mouseMode'] == pg.ViewBox.RectMode:
                    if ev.isFinish():  ## This is the final move in the drag; change the view scale now
                        #print "finish"
                        self.rbScaleBox.hide()
                        ax = QtCore.QRectF(Point(ev.buttonDownPos(ev.button())), Point(pos))
                        ax = self.childGroup.mapRectFromParent(ax)
                        self.showAxRect(ax)
                        self.axHistoryPointer += 1
                        self.axHistory = self.axHistory[:self.axHistoryPointer] + [ax]
                    else:
                        ## update shape of scale box
                        self.updateScaleBox(ev.buttonDownPos(), ev.pos())
                else:
                    tr = dif*mask
                    tr = self.mapToView(tr) - self.mapToView(Point(0,0))
                    x = tr.x() if mask[0] == 1 else None
                    y = tr.y() if mask[1] == 1 else None

                    self._resetTarget()
                    if x is not None or y is not None:
                        self.translateBy(x=x, y=y)
                    self.sigRangeChangedManually.emit(self.state['mouseEnabled'])

# class ImageDraw(pg.ImageItem):
#     """
#     **Bases:** :class:`GraphicsObject <pyqtgraph.GraphicsObject>`
#     GraphicsObject displaying an image. Optimized for rapid update (ie video display).
#     This item displays either a 2D numpy array (height, width) or
#     a 3D array (height, width, RGBa). This array is optionally scaled (see
#     :func:`setLevels <pyqtgraph.ImageItem.setLevels>`) and/or colored
#     with a lookup table (see :func:`setLookupTable <pyqtgraph.ImageItem.setLookupTable>`)
#     before being displayed.
#     ImageItem is frequently used in conjunction with
#     :class:`HistogramLUTItem <pyqtgraph.HistogramLUTItem>` or
#     :class:`HistogramLUTWidget <pyqtgraph.HistogramLUTWidget>` to provide a GUI
#     for controlling the levels and lookup table used to display the image.
#     """

#     sigImageChanged = QtCore.pyqtSignal()

#     def __init__(self, image=None, viewbox=None, parent=None, **kargs):
#         super(ImageDraw, self).__init__()
#         #self.image=None
#         self.viewbox=viewbox
#         self.levels = np.array([0,255])
#         self.lut = None
#         self.autoDownsample = False
#         self.axisOrder = 'row-major'
#         self.removable = False

#         self.parent = parent
#         #kernel[1,1] = 1
#         # self.setDrawKernel(kernel_size=self.parent.brush_size)
#         self.parent.current_stroke = []
#         self.parent.in_stroke = False
        

#     def create_start(self, pos):
#         self.scatter = pg.ScatterPlotItem([pos.x()], [pos.y()], pxMode=False,
#                                         pen=pg.mkPen(color=(255,0,0), width=self.parent.brush_size),
#                                         size=max(3*2, self.parent.brush_size*1.8*2), brush=None)
#         self.parent.p0.addItem(self.scatter)

#     def is_at_start(self, pos):
#         thresh_out = max(6, self.parent.brush_size*3)
#         thresh_in = max(3, self.parent.brush_size*1.8)
#         # first check if you ever left the start
#         if len(self.parent.current_stroke) > 3:
#             stroke = np.array(self.parent.current_stroke)
#             dist = (((stroke[1:,1:] - stroke[:1,1:][np.newaxis,:,:])**2).sum(axis=-1))**0.5
#             dist = dist.flatten()
#             #print(dist)
#             has_left = (dist > thresh_out).nonzero()[0]
#             if len(has_left) > 0:
#                 first_left = np.sort(has_left)[0]
#                 has_returned = (dist[max(4,first_left+1):] < thresh_in).sum()
#                 if has_returned > 0:
#                     return True
#                 else:
#                     return False
#             else:
#                 return False

#     def end_stroke(self):
#         self.parent.p0.removeItem(self.scatter)
#         if not self.parent.stroke_appended:
#             self.parent.strokes.append(self.parent.current_stroke)
#             self.parent.stroke_appended = True
#             self.parent.current_stroke = np.array(self.parent.current_stroke)
#             ioutline = self.parent.current_stroke[:,3]==1
#             self.parent.current_point_set.extend(list(self.parent.current_stroke[ioutline]))
#             self.parent.current_stroke = []
#             if self.parent.autosave:
#                 self.parent.add_set()
#         if len(self.parent.current_point_set) > 0 and self.parent.autosave:
#             self.parent.add_set()
#         self.parent.in_stroke = False

#     def drawAt(self, pos, ev=None):
#         mask = self.strokemask
#         set = self.parent.current_point_set
#         stroke = self.parent.current_stroke
#         pos = [int(pos.y()), int(pos.x())]
#         dk = self.drawKernel
#         kc = self.drawKernelCenter
#         sx = [0,dk.shape[0]]
#         sy = [0,dk.shape[1]]
#         tx = [pos[0] - kc[0], pos[0] - kc[0]+ dk.shape[0]]
#         ty = [pos[1] - kc[1], pos[1] - kc[1]+ dk.shape[1]]
#         kcent = kc.copy()
#         if tx[0]<=0:
#             sx[0] = 0
#             sx[1] = kc[0] + 1
#             tx    = sx
#             kcent[0] = 0
#         if ty[0]<=0:
#             sy[0] = 0
#             sy[1] = kc[1] + 1
#             ty    = sy
#             kcent[1] = 0
#         if tx[1] >= self.parent.Ly-1:
#             sx[0] = dk.shape[0] - kc[0] - 1
#             sx[1] = dk.shape[0]
#             tx[0] = self.parent.Ly - kc[0] - 1
#             tx[1] = self.parent.Ly
#             kcent[0] = tx[1]-tx[0]-1
#         if ty[1] >= self.parent.Lx-1:
#             sy[0] = dk.shape[1] - kc[1] - 1
#             sy[1] = dk.shape[1]
#             ty[0] = self.parent.Lx - kc[1] - 1
#             ty[1] = self.parent.Lx
#             kcent[1] = ty[1]-ty[0]-1


#         ts = (slice(tx[0],tx[1]), slice(ty[0],ty[1]))
#         ss = (slice(sx[0],sx[1]), slice(sy[0],sy[1]))
#         self.image[ts] = mask[ss]

#         for ky,y in enumerate(np.arange(ty[0], ty[1], 1, int)):
#             for kx,x in enumerate(np.arange(tx[0], tx[1], 1, int)):
#                 iscent = np.logical_and(kx==kcent[0], ky==kcent[1])
#                 stroke.append([self.parent.currentZ, x, y, iscent])
#         self.updateImage()

#     def setDrawKernel(self, kernel_size=3):
#         bs = kernel_size
#         kernel = np.ones((bs,bs), np.uint8)
#         self.drawKernel = kernel
#         self.drawKernelCenter = [int(np.floor(kernel.shape[0]/2)),
#                                  int(np.floor(kernel.shape[1]/2))]
#         onmask = 255 * kernel[:,:,np.newaxis]
#         offmask = np.zeros((bs,bs,1))
#         opamask = 100 * kernel[:,:,np.newaxis]
#         self.redmask = np.concatenate((onmask,offmask,offmask,onmask), axis=-1)
#         self.strokemask = np.concatenate((onmask,offmask,onmask,opamask), axis=-1)


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
    
        # save state here initiallly
        if self._canDraw(event) and not len(self.parent.undo_stack):
            self.parent.update_layer(save_state=True)
            
        x, y = int(event.pos().x()), int(event.pos().y())
        

        # Safely check for pick_label_enabled and flood_fill_enabled
        if getattr(self.parent, 'pick_label_enabled', False):
            self._pickLabel(x, y)
            event.accept()
        elif getattr(self.parent, 'flood_fill_enabled', False):
            self._floodFill(x, y, getattr(self.parent, 'current_label', 0))  # Default label to 0
            event.accept()
        elif self._canDraw(event):
            self._drawing = True
            self._lastPos = event.pos()
            self._drawPixel(x, y) 
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drawing:
            event.accept()
            self._paintLine(self._lastPos, event.pos())
            self._lastPos = event.pos()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._drawing:
            event.accept()
            self._paintLine(self._lastPos, event.pos())
            self._drawing = False
            self._lastPos = None
            # self.parent.update_layer(save_state=True) # save state after completion
        else:
            super().mouseReleaseEvent(event)
            
    def _canDraw(self, event):
        """Checks if conditions allow drawing instead of panning."""
        # If brush_size is 0 or space is pressed, do not draw
        if getattr(self.parent, 'brush_size', 0) == 0 or (not self.parent.SCheckBox.isChecked()):
            return False
        if getattr(self.parent, 'spacePressed', False):
            return False
        # Must be left mouse
        if event.button() != QtCore.Qt.LeftButton and not (event.buttons() & QtCore.Qt.LeftButton):
            return False
        return True

    def _paintAt(self, pos):
        """Draw a single point (useful for initial press)."""
        self._drawPixel(pos.x(), pos.y())

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

    def _drawPixel(self, x, y):
        """Draws a circular area or single pixel using a precomputed kernel."""
        brush_size = getattr(self.parent, 'brush_size', 1)
        label = getattr(self.parent, 'current_label', 0)
        z = self.parent.currentZ

        array = self.parent.cellpix[z]
        height, width = array.shape

        # If brush size is 1, draw a single pixel
        if brush_size == 1:
            if 0 <= y < height and 0 <= x < width:
                array[int(y), int(x)] = label
        else:
            # Ensure the kernel is up to date
            if not hasattr(self, '_kernel') or self._kernel.shape[0] != 2 * brush_size + 1:
                self._generateKernel(brush_size)

            kernel = self._kernel
            kernel_radius = kernel.shape[0] // 2

            # Define bounds for applying the kernel
            x_min = max(0, int(x) - kernel_radius)
            x_max = min(width, int(x) + kernel_radius + 1)
            y_min = max(0, int(y) - kernel_radius)
            y_max = min(height, int(y) + kernel_radius + 1)

            # Define bounds within the kernel
            kx_min = max(0, kernel_radius - int(x))
            kx_max = kernel_radius + (x_max - int(x))
            ky_min = max(0, kernel_radius - int(y))
            ky_max = kernel_radius + (y_max - int(y))

            # Apply the kernel to the array
            array[y_min:y_max, x_min:x_max][kernel[ky_min:ky_max, kx_min:kx_max]] = label

        self.parent.update_layer()  # Refresh the display
        
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
        show_histogram = True
        
    def mousePressEvent(self, event):
        if self.show_histogram:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.show_histogram:
            super().mouseMoveEvent(event)

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
        for pen in [pen]: #get rid of dirst entry, shadow of some sort
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

