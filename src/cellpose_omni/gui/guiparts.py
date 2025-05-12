from cellpose_omni.gui import logger # from __init__.py in parent directory

from PyQt6 import QtGui, QtCore
from PyQt6.QtWidgets import QAbstractItemView
from PyQt6.QtGui import QPainter
from PyQt6.QtWidgets import QApplication, QRadioButton, QWidget, QDialog, QButtonGroup, QSlider, QStyle, QStyleOptionSlider, QGridLayout, QPushButton, QLabel, QLineEdit, QDialogButtonBox, QComboBox
import pyqtgraph as pg
from pyqtgraph.Point import Point
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


from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QTableWidget,
    QTableWidgetItem, QPushButton, QHeaderView, QSizePolicy
)

class MyTableWidget(QTableWidget):
    def __init__(self, parentDock):
        super().__init__()
        self.parentDock = parentDock

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            row = self.currentRow()
            col = self.currentColumn()
            if col < self.columnCount() - 1:
                # Move to the next column in this row
                super().keyPressEvent(event)
                self.setCurrentCell(row, col + 1)
            else:
                # Last column -> create a new row
                super().keyPressEvent(event)
                self.parentDock.addRow()
                self.setCurrentCell(row + 1, 0)
        else:
            super().keyPressEvent(event)

from PyQt6.QtWidgets import QStyledItemDelegate
from PyQt6.QtGui import QPen, QColor
from PyQt6.QtCore import Qt

class InnerGridDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        # First, let the default delegate paint the cell
        super().paint(painter, option, index)
        
        # Determine the model's dimensions for this cell
        model = index.model()
        row = index.row()
        col = index.column()
        
        pen = QPen(QColor(128,128,128,128))  # RGB (136,136,136) with alpha=50 (out of 255)
        # pen = QPen(QColor("#88888811"))  # Light gray, semi-transparent
        # pen = pg.mkPen(self.accent)

        painter.setPen(pen)
        
        # Adjust the cell rect slightly for pixel alignment
        # rect = option.rect.adjusted(0, 0, -0.5, -0.5)
        rect = QtCore.QRectF(option.rect).adjusted(0, 0, -0.5, -0.5)
        
        # Draw vertical line on the right edge (if not last column)
        if col < model.columnCount() - 1:
            painter.drawLine(rect.topRight(), rect.bottomRight())
        
        # Draw horizontal line on the bottom edge (if not last row)
        if row < model.rowCount() - 1:
            painter.drawLine(rect.bottomLeft(), rect.bottomRight())

class LinksDock(QDockWidget):
    def __init__(self, links=None, parent=None):
        super().__init__("Links Editor", parent)
        self._links = list(links) if links else []
        
    
        container = QWidget()
        layout = QVBoxLayout()
        container.setLayout(layout)
        self.setWidget(container)

        # Let the dock expand in the main window as needed
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Create the table
        self.table = MyTableWidget(self)
        self.table.setRowCount(max(3, len(self._links)))
        self.table.setColumnCount(2)

        # Optionally hide row/column headers
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setVisible(False)
        # Hide any text in the headers (e.g. no row numbers, no column titles)
        # Force them to a very small size so they don’t show text
  
        
        
        # Make columns auto‐stretch
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        # Let the table fill the dock’s area
        self.table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # self.table.setFrameShape(QFrame.Shape.NoFrame)  # turn off the default table frame
        self.table.setShowGrid(False)
        self.table.setFrameShape(QFrame.Shape.NoFrame)   # Turn off the outer frame
        self.table.setItemDelegate(InnerGridDelegate())
        
        # Make row selection the default
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

        # Enable alternating row colors, and update the stylesheet accordingly:
        alternating_color = "rgba(128, 128, 128, 0.1)"  
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet(f"""
            QTableWidget {{
                /* Leave the normal background unchanged (inherited from theme) */
                alternate-background-color: {alternating_color};
            }}
        """)
        
        # Populate the table from _links
        for row, (valA, valB) in enumerate(self._links):
            if row >= self.table.rowCount():
                break
            self._set_item(row, 0, str(valA))
            self._set_item(row, 1, str(valB))

        layout.addWidget(self.table)

        # Button to append rows
        self.btnAddRow = QPushButton("Add Row")
        self.btnAddRow.clicked.connect(self.addRow)
        layout.addWidget(self.btnAddRow)

        # Button to delete selected rows
        self.btnRemoveRow = QPushButton("Delete Row")
        self.btnRemoveRow.clicked.connect(self.removeSelectedRows)
        layout.addWidget(self.btnRemoveRow)

        # Track edits
        self.table.itemChanged.connect(self.on_item_changed)
        
        

    def _set_item(self, row, col, text):
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.setItem(row, col, item)

    def addRow(self):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self._set_item(row, 0, "")
        self._set_item(row, 1, "")

    def removeSelectedRows(self):
        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            return
        for sel in reversed(selected_rows):
            row = sel.row()
            self.table.removeRow(row)
            if row < len(self._links):
                self._links.pop(row)

    def on_item_changed(self, changed_item):
        row = changed_item.row()
        itemA = self.table.item(row, 0)
        itemB = self.table.item(row, 1)
        if itemA and itemB and itemA.text().isdigit() and itemB.text().isdigit():
            valA = int(itemA.text())
            valB = int(itemB.text())
            while len(self._links) <= row:
                self._links.append((0, 0))
            self._links[row] = (valA, valB)

    def getLinks(self):
        return self._links
 
# not used?       
# class NoMouseFilter(QObject):
#     def eventFilter(self, obj, event):
#         if event.type() in (QEvent.Type.GraphicsSceneMousePress,
#                             QEvent.Type.GraphicsSceneMouseMove,
#                             QEvent.Type.GraphicsSceneMouseRelease):
#             return True  # Consume the event.
#         return super().eventFilter(obj, event)

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
        parent.viewbox.setXRange(self.xrange[0], self.xrange[1])
        parent.viewbox.setYRange(self.yrange[0], self.yrange[1])
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
            button.toggled.connect(self.btnpress)
            self.parent.l0.addWidget(button, row+b,col,1,2)
        self.setExclusive(True)
        #self.buttons.

    def btnpress(self):
        b = self.checkedId()
        self.parent.cell_type = b

class ViewRadioButtons(QButtonGroup):
    def __init__(self, parent=None, row=0, col=0):
        super(ViewRadioButtons, self).__init__()
        parent.color = 0
        self.parent = parent
        self.bstr = ["image", "flow field", "distance field", "boundary logits", "affinity sum"]
        #self.buttons = QButtonGroup()
        self.dropdown = []
        for b in range(len(self.bstr)):
            button = QRadioButton(self.bstr[b])
            # button.setStyleSheet('color: white;')
            button.setFont(parent.medfont)
            if b==0:
                button.setChecked(True)
            self.addButton(button, b)
            button.clicked.connect(self.btnpress) #"toggled" sends update twice
            self.parent.l0.addWidget(button, row+b,col,1,2)
        # self.setExclusive(True)
        #self.buttons.

    def btnpress(self):
        b = self.checkedId()
        self.parent.view = b
        
        # set some defaults
        # if b in (1,2,3,4):
        isImg = b==0
        self.parent.outlinesOn = isImg
        self.parent.OCheckBox.setChecked(isImg)

        
        if self.parent.loaded:
            # self.parent.update_layer()
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


    def _intpos(self, event):
        pos = event.pos()
        x, y = int(pos.x()), int(pos.y())
        return x, y
        
    def mousePressEvent(self, event):
    
        x, y = self._intpos(event)

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
            self._lastPos = self._intpos(event)
            self._drawPixel(x, y) 
            # self._paintLine(self._lastPos, event.pos()) # test
            event.accept()
        else:
            pg.ImageItem.mousePressEvent(self, event)
            
    def mouseMoveEvent(self, event):
        if self._drawing:
            pos = self._intpos(event)
            self._paintLine(self._lastPos, pos)
            self._lastPos = pos
            event.accept()
        else:
            pg.ImageItem.mouseMoveEvent(self, event)
            

    def mouseReleaseEvent(self, event):
        if self._drawing:
            # self._paintLine(self._lastPos, event.pos())
            x, y = self._intpos(event)
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
        if not self.parent.PencilCheckBox.isChecked():
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
        # x0, y0 = int(startPos.x()), int(startPos.y())
        # x1, y1 = int(endPos.x()),   int(endPos.y())
        x0, y0 = startPos
        x1, y1 = endPos        

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
        masks = self.parent.mask_stack[z]
        bounds = self.parent.outl_stack[z] # this might be a direct hook to the display
        
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
        # If the computed region has no area, skip drawing.
        if r1 <= r0 or c1 <= c0:
            return

        # Kernel slice
        kernel = self._kernel
        kr0, kr1 = max(0, kr - y), kr + (r1 - y)
        kc0, kc1 = max(0, kr - x), kr + (c1 - x)
        if kr1 <= kr0 or kc1 <= kc0:
            return
        ker_slc = (slice(kr0, kr1), slice(kc0, kc1))
        
        # similar to _get_affinity, should check that too for some factoring out
        # source inds restricts to valid sources 
        source_indices = self.parent.ind_matrix[arr_slc][kernel[ker_slc]]
        # source_indices = source_indices[source_indices>-1]
        source_coords = tuple(c[source_indices] for c in self.parent.coords)
        targets = []
        
        
        # Dilated slice (expand by 1 in all directions)
        d = 1
        dil_r0, dil_r1 = max(0, r0 - d), min(r1 + d, height)
        dil_c0, dil_c1 = max(0, c0 - d), min(c1 + d, width)
        dil_slc = (slice(dil_r0, dil_r1), slice(dil_c0, dil_c1))


        # print('fff\n', self.parent.coords[0].shape, np.prod(masks.shape), '\n')
        
        masks[source_coords] = label # apply it here
        
        if np.any(source_indices==-1):
            print('\n ERROR, negative index in source_indices')

        steps = self.parent.steps
        dim = self.parent.dim
        idx = self.parent.inds[0][0]
        shape = self.parent.shape
        # print('affinity none?', affinity is None)
        # print('affinity shape', affinity.shape, np.any(affinity))
        # print('source',source_coords)
        # print('shape', shape)
        

        if affinity is not None and affinity.shape[-dim:] == masks.shape:
            for i in self.parent.non_self:
                step = steps[i]
                # target_coords = tuple(np.clip(c+s,0,l-1) for c,s,l in zip(source_coords, step, shape))
                # print('A', np.array(target_coords).shape)
                

   
                # neigh_indices = self.parent.neigh_inds[i]

                # # sel = neigh_indices>-1 # non-foreground pixels have index -1, and that would mess up indexing
                # # source_inds = self.parent.indexes[sel] # we therefore only deal with source pixels that have a valid target 
                # # target_inds = neigh_indices[source_inds] # and these are the corresponding valid targets 
                
                # print('neight', neigh_indices[source_indices], source_indices)
                # target_coords = tuple(self.parent.neighbors[:,i,source_indices])

                # print('B',target_coords, masks[target_coords], label)
                
                # must remove self references 
                # could check if the source is equal to target
                # or do it loke in other ares, where i check for validdity with indexing matrix
                # or I can do it like in get_affinity_torch, where I simply slice in each direction as needed
                # the only real reason to do it with inds is becasue of the non-spatial affinity 
                # witht he spatial array, it is simple to just avoid out of bounds with slicing to -1
                # however, this apprach is using coords, not slices 
                
                tgt_coords = []
                src_coords = []
                valid = np.ones_like(source_coords[0])
                for c,s,l in zip(source_coords, step, shape):
                    tc = c+s
                    is_valid = np.logical_and(tc>=0, tc<=l-1)
                    valid = is_valid if valid is None else np.logical_and(valid, is_valid)
                    tgt_coords.append(tc)
                    src_coords.append(c)
                        
                if np.any(valid):
                    tgt_coords = tuple(c[valid] for c in tgt_coords)
                    src_coords = tuple(c[valid] for c in src_coords)
                                        
                    target_labels = masks[tgt_coords]
                    if label!=0:
                        connect = target_labels==label
                    else:
                        connect = False
                    
                    # update affinity graph
                    affinity[i][src_coords] = affinity[-(i+1)][tgt_coords] = connect                
                    targets.append(tgt_coords)
                    
        else:
            print('affinity graph not initialized',affinity.shape[-dim:], masks.shape) 
               
    
        # define a region around the source and target pixels
        targets.append(source_coords)
        
        surround_coords = tuple(np.concatenate(arrays, axis=0) for arrays in zip(*targets))

        

# !!! while drawing, could enable a mode not to connect to self unless it is to the direction opposite motion, so tht we can draw disconnected snakes 


        
        update_inds = []
        update_alpha = []
        # have to wait to update affinity after looping over all directions   
        for i in self.parent.non_self[:idx]:
            step = steps[i]
            target_coords = tuple(np.clip(c+s,0,l-1) for c,s,l in zip(source_coords, step, shape))
            opp_target_coords = tuple(np.clip(c-s,0,l-1) for c,s,l in zip(source_coords, step, shape))

            inds = self.parent.pixelGridOverlay.lineIndices[source_coords + (i,)].tolist() 
            inds += self.parent.pixelGridOverlay.lineIndices[opp_target_coords+(i,)].tolist()

            update_inds += inds
            update_alpha.append(affinity[i][source_coords])
            update_alpha.append(affinity[i][opp_target_coords])
                    
        self.parent.pixelGridOverlay.toggle_lines_batch(update_inds, 
                                                      np.concatenate(update_alpha), 
                                                      visible=False)
    
                
        # I think I am missing the background pixels connected to it
        # maybe the cleanup will do it 

        # I could add a cleanup step here from get_affinity_torch
        # it could operate on just the dilated region to update the affinity graph
                 
    

        
        # some strangeness with outlines on and masks off, as if the masks are zero?
                    
        # update boundary
        if len(surround_coords):
            # print('tt',len(targets), targets)
            # print('sc',surround_coords)
            bd = affinity_to_boundary(masks[surround_coords], affinity[(Ellipsis,)+surround_coords], None, dim=dim)
            bd = bd*masks[surround_coords]
            bounds[surround_coords] = bd
        
        
        # also take care of any orphaned masks
        # print(affinity[:,*surround_coords].sum(axis=0).shape, masks[surround_coords].shape)
        
        # masks[*surround_coords][affinity[:,*surround_coords].sum(axis=0)==0] = 0
        # masks[*surround_coords] *= affinity[:,*surround_coords].sum(axis=0)>0
        # masks[surround_coords] *= affinity[(Ellipsis,)+surround_coords].sum(axis=0)>0
        
        # num_new_pixels = len(surround_coords[0])
        # print('num_new_pixels', num_new_pixels)
        # if label == 0:
        
        # could limit the sum to over the cardinal directions
        # such that we just dissallow vertex connections
        # if cleanup:
        #     self._cleanup()
        # masks[surround_coords] *= affinity[(Ellipsis,)+surround_coords].sum(axis=0) > 0

            
        # print('info', self.parent.pixelGridOverlay.lineIndices.shape)
        # Update only the affected region of the overlay
        # self.parent.draw_layer(region=(c0, c1, r0, r1), z=z)
        # pass a dilated region to catch the outer edges of the new boundary 
        self.parent.draw_layer(region=(dil_c0, dil_c1, dil_r0, dil_r1), z=z)
    
    
    def _cleanup(self):
        # brush_size = getattr(self.parent, 'brush_size', 1)
        # label = getattr(self.parent, 'current_label', 0)
        z = self.parent.currentZ
        masks = self.parent.mask_stack[z]
        bounds = self.parent.outl_stack[z] # this might be a direct hook to the display
        
        affinity = self.parent.affinity_graph
        height, width = masks.shape
        
        steps = self.parent.steps
        dim = self.parent.dim
        idx = self.parent.inds[0][0]
  
        D = dim
        S = len(steps)
        cutoff = 3**(D-1) 

        source_slices, target_slices = [[[[] for _ in range(D)] for _ in range(S)] for _ in range(2)]


        s1,s2,s3 = slice(1,None), slice(0,-1), slice(None,None) # this needs to be generalized to D dimensions
        for i in range(S):
            for j in range(D):
                s = steps[i][j]
                target_slices[i][j], source_slices[i][j] = (s1,s2) if s>0 else (s2,s1) if s<0 else (s3,s3)
                
                    
    
        csum = np.sum(affinity,axis=0)
        print('max', np.unique(csum), 'cutoff', cutoff)
        keep = csum>=cutoff

        idx = self.parent.inds[0][0]

        for i in self.parent.non_self:#:idx]:

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
            
            
            # inds = self.parent.pixelGridOverlay.lineIndices[*target_coords,i].tolist() # 3.11+ for this syntax
            inds = self.parent.pixelGridOverlay.lineIndices[target_coords+(i,)].tolist()
            
            opp_target_coords = tuple(np.clip(c-s,0,l-1) for c,s,l in zip(source_coords, step,[Ly,Lx]))
            # inds += self.parent.pixelGridOverlay.lineIndices[*opp_target_coords,i].tolist()
            inds += self.parent.pixelGridOverlay.lineIndices[opp_target_coords+(i,)].tolist()

            update_inds += inds
            update_alpha.append(affinity[i][source_coords])
            update_alpha.append(affinity[i][opp_target_coords])
                    
        self.parent.pixelGridOverlay.toggle_lines_batch(update_inds, 
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
        
        # print('yoyo', masks.ndim, dim,  self.parent.mask_stack.ndim)
        bounds = bd
        
        
        # also take care of any orphaned masks
        # print(affinity[:,*surround_coords].sum(axis=0).shape, masks[surround_coords].shape)
        
        # masks[*surround_coords][affinity[:,*surround_coords].sum(axis=0)==0] = 0
        masks *= affinity.sum(axis=0)>0
        

        # Update only the affected region of the overlay
        # self.parent.draw_layer(region=(c0, c1, r0, r1), z=z)
        # pass a dilated region to catch the outer edges of the new boundary 
        self.parent.draw_layer(z=z)

    def _floodFill(self, x, y, label):        
        """Perform flood fill starting at (x, y) with the given label."""
        z = self.parent.currentZ
        masks = self.parent.mask_stack[z]
        bounds = self.parent.outl_stack[z] # this might be a direct hook to the display
        
        
        old_label = masks[int(y), int(x)]
        
        if old_label == label:
            return  # Nothing to change
    
        affinity = self.parent.affinity_graph
        if affinity is None:
            return
    
    
        # Use an efficient queue-based flood fill
        stack = [(int(y), int(x))]
        affected_pixels = set()
        while stack:
            cy, cx = stack.pop()
            if 0 <= cy < masks.shape[0] and 0 <= cx < masks.shape[1] and masks[cy, cx] == old_label:
                masks[cy, cx] = label
                affected_pixels.add((cy, cx))
                # Add neighbors to the stack
                stack.extend([(cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)])
    
        # Define a region around the affected pixels
        affected_coords = np.array(list(affected_pixels))
        r0, r1 = affected_coords[:, 0].min(), affected_coords[:, 0].max() + 1
        c0, c1 = affected_coords[:, 1].min(), affected_coords[:, 1].max() + 1
        dil_r0, dil_r1 = max(0, r0 - 3), min(r1 + 3, masks.shape[0])
        dil_c0, dil_c1 = max(0, c0 - 3), min(c1 + 3, masks.shape[1])
        dil_slc = (slice(dil_r0, dil_r1), slice(dil_c0, dil_c1))
    
    
        source_coords = tuple(affected_coords.T)
        targets = []        
        masks[source_coords] = label # apply it here


        steps = self.parent.steps
        shape = self.parent.shape
        dim = self.parent.dim
        idx = self.parent.inds[0][0]

        if affinity is not None and affinity.shape[-dim:] == masks.shape:
            for i in self.parent.non_self:
                step = steps[i]
                # replace with neighhbors array
                # print('target n', self.parent.neighbors.shape, np.array(target_coords).shape)
                target_coords = tuple(np.clip(c+s,0,l-1) for c,s,l in zip(source_coords, step, shape))

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
               
    
        # this block should be factored out 
        
        # define a region around the source and target pixels
        targets.append(source_coords)
        surround_coords = tuple(np.concatenate(arrays, axis=0) for arrays in zip(*targets))
        
        update_inds = []
        update_alpha = []
        # have to wait to update affinity after looping over all directions   
        for i in self.parent.non_self[:idx]:
            step = steps[i]
            target_coords = tuple(np.clip(c+s,0,l-1) for c,s,l in zip(source_coords, step, shape))
            opp_target_coords = tuple(np.clip(c-s,0,l-1) for c,s,l in zip(source_coords, step, shape))
            
            
            inds = self.parent.pixelGridOverlay.lineIndices[source_coords +(i,)].tolist() # need 3.11 for this syntax?
            inds += self.parent.pixelGridOverlay.lineIndices[opp_target_coords+(i,)].tolist()
            
            update_inds += inds
            update_alpha.append(affinity[i][source_coords])
            update_alpha.append(affinity[i][opp_target_coords])
                    
        self.parent.pixelGridOverlay.toggle_lines_batch(update_inds, 
                                                      np.concatenate(update_alpha), 
                                                      visible=True)

                    
        # update boundary
        bd = affinity_to_boundary(masks[surround_coords], affinity[(Ellipsis,)+surround_coords], None, dim=dim)
        bd = bd*masks[surround_coords]
        bounds[surround_coords] = bd
        masks[surround_coords] *= affinity[(Ellipsis,)+surround_coords].sum(axis=0)>0
        self.parent.draw_layer(region=(dil_c0, dil_c1, dil_r0, dil_r1), z=z)
        
        #looks like another call might be redrawing the whole layer 
        # self.parent.update()
        # self.parent.draw_layer(region=(dil_c0, dil_c1, dil_r0, dil_r1), z=z)
    
    def _pickLabel(self, x, y):
        """Set the current label to the value under the cursor."""
        z = self.parent.currentZ
        self.parent.current_label = self.parent.mask_stack[z, int(y), int(x)]
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
    # sigLookupTableChanged = QtCore.pyqtSignal(object)
    # sigLevelsChanged = QtCore.pyqtSignal(object)
    # sigLevelChangeFinished = QtCore.pyqtSignal(object)

    def __init__(self, image=None, fillHistogram=True, levelMode='mono',
                 gradientPosition='right', orientation='vertical'):
        super().__init__(
            image=image,
            fillHistogram=fillHistogram,
            levelMode=levelMode,
            gradientPosition=gradientPosition,
            orientation=orientation
        )
        self._orig_mousePressEvent = super().mousePressEvent
        self._orig_mouseMoveEvent = super().mouseMoveEvent
        self._orig_mouseReleaseEvent = super().mouseReleaseEvent
        self._orig_paint = super().paint
        

        self.imageitem = image
        self._view = 0
        # Make sure there's a place to store full item states for each view.
        if not hasattr(self, 'view_states'):
            self.view_states = {}
            

        # Connect signals for color stops...
        if hasattr(self.gradient, 'sigGradientChanged'):
            self.gradient.sigGradientChanged.connect(self._on_gradient_changed)
        elif hasattr(self.gradient, 'sigGradientChangeFinished'):
            self.gradient.sigGradientChangeFinished.connect(self._on_gradient_changed)

        # ...and region/levels so we capture user changes to min/max handles.
        self.region.sigRegionChanged.connect(self._on_region_changed)
        self.region.sigRegionChangeFinished.connect(self._on_region_changed)
    
    # def paint(self, painter, option, widget=None):
    #     old_brush = self.gradient.gradRect.brush()      
    #     if getattr(self, 'discrete_mode', False):
    #         # In discrete mode, skip the default overlay drawing.
    #         # return
    #         self.gradient.gradRect.setBrush(QtGui.QBrush(QtCore.Qt.BrushStyle.NoBrush))
    #     # Otherwise, call the original painting.
    #     self._orig_paint(painter, option, widget)
    #     self.gradient.gradRect.setBrush(old_brush)
    #     self.gradient.backgroundRect.hide() # the cause of the hatching pattern 
        
    
    def set_view(self, v, preset=None, default_cmaps=None):
        self._view = v
        state = self.view_states.get(v)  # This is the *full item* state, not just gradient.
            
        if state is not None:
            self.restoreState(state)  # calls HistogramLUTItem.restoreState(state)
        else:
            self.autoRange()

            # No stored state for this view => optionally load a preset
            if preset is not None:
                self.gradient.loadPreset(preset)
                
                if v != 0 and default_cmaps and (preset in default_cmaps):
                    # If we want alpha=0 on the first tick for non-zero views:
                    state = self.gradient.saveState()  # Just the gradient sub-dict
                    if 'ticks' in state:
                        pos, color = state['ticks'][0]
                        color = list(color)
                        color[3] = 0
                        state['ticks'][0] = [pos, tuple(color)]
                    # Re‐apply that sub‐dict to the item’s gradient
                    self.gradient.restoreState(state)

            # Then store the entire item’s state now that it’s configured
            self._store_current_state()

    def setDiscreteMode(self, discrete: bool, n_bands: int = 5):
        """
        Toggle the gradient's mode between discrete and continuous.
        When discrete is True, override the gradient's paint method with a custom
        _discrete_gradient_paint method that draws discrete color blocks.
        Also disable any caching and hide gradRect so that only the discrete drawing is visible.
        When False, restore the original paint method and show gradRect.
        """
        self.discrete_mode = discrete
        if discrete:
            # Force the gradient to RGB mode.
            st = self.gradient.saveState()
            if st.get('mode', '').lower() == 'hsv':
                st['mode'] = 'rgb'
                self.gradient.restoreState(st)
            # Save original paint method if not already saved.  
            if not hasattr(self.gradient, '_original_paint'):
                self.gradient._original_paint = self.gradient.paint
            # Store the number of discrete bands.
            self.gradient._discrete_n_bands = n_bands
            # Set a pointer to the parent HistLUT instance.
            self.gradient._parent_histlut = self
            # Override the paint method with our custom discrete painter.
            self.gradient.paint = self._discrete_gradient_paint.__get__(self.gradient, type(self.gradient))
            # Hide the continuous gradient: disable gradRect painting and caching.
            self.gradient.gradRect.hide()

        else:
            if hasattr(self.gradient, '_original_paint'):
                self.gradient.paint = self.gradient._original_paint
            self.gradient.gradRect.show()

        self.gradient.update()
        

    def _discrete_gradient_paint(self, painter, opt, widget=None):
        """
        Custom paint method for a gradient item that draws discrete color bands
        without gaps between them. Each band is drawn as a solid rectangle using
        integer boundaries computed to exactly fill the drawing area, with a 1-pixel overlap
        between adjacent bands to cover any rounding gaps.
        """
        painter.save()
        painter.setPen(Qt.NoPen)
        # painter.setPen(self.tickPen)
        
        
        # Get the drawing rectangle from gradRect as an integer QRect.
        rect = self.gradRect.rect().toRect()
        nBands = getattr(self, '_discrete_n_bands', 5)
        
        if rect.width() >= rect.height():
            # Horizontal orientation: divide width exactly among bands.
            total_width = rect.width()
            base_width = total_width // nBands
            remainder = total_width % nBands
            x = rect.left()
            for i in range(nBands):
                extra = 1 if i < remainder else 0
                w = base_width + extra
                # For all bands except the last, add 1 pixel to the right boundary to force overlap.
                if i < nBands - 1:
                    bandRect = QtCore.QRect(x, rect.top(), w + 1, rect.height())
                else:
                    bandRect = QtCore.QRect(x, rect.top(), w, rect.height())
                # frac = (i + 0.5) / nBands
                frac = (i ) / (nBands-1)                
                
                qcol = self._parent_histlut._sample_colormap(frac)
                painter.setBrush(QtGui.QBrush(qcol, Qt.BrushStyle.SolidPattern))
                painter.drawRect(bandRect)
                x += w
        else:
            # Vertical orientation: divide height exactly among bands.
            total_height = rect.height()
            base_height = total_height // nBands
            remainder = total_height % nBands
            y = rect.top()
            for i in range(nBands):
                extra = 1 if i < remainder else 0
                h = base_height + extra
                if i < nBands - 1:
                    bandRect = QtCore.QRect(rect.left(), y, rect.width(), h + 1)
                else:
                    bandRect = QtCore.QRect(rect.left(), y, rect.width(), h)
                frac = (i + 0.5) / nBands
                qcol = self._parent_histlut._sample_colormap(frac)
                painter.setBrush(QtGui.QBrush(qcol, Qt.BrushStyle.SolidPattern))
                painter.drawRect(bandRect)
                y += h
        
        # drow the border 
        pen = self.tickPen
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawRect(rect)

        painter.restore()

                
    def _sample_colormap(self, fraction: float) -> QtGui.QColor:
        """
        Sample the current colormap at the given fraction (0 to 1) and return a fully opaque QColor.
        Instead of using .map(), this uses the lookup table directly.
        """
        # Force the gradient into RGB mode every time.
        st = self.gradient.saveState()
        if st.get('mode', '').lower() == 'hsv':
            st['mode'] = 'rgb'
            self.gradient.restoreState(st)
        
        lut = self.gradient.colorMap().getLookupTable(nPts=256)
        idx = int(round(fraction * 255))
        idx = max(0, min(idx, 255))
        entry = lut[idx]
        if len(entry) == 3:
            r, g, b = entry
            a = 255
        else:
            r, g, b, a = entry
        return QtGui.QColor(r, g, b, a)

    def _on_gradient_changed(self):
        # Called when the user moves color stops in the gradient
        self._store_current_state()

    def _on_region_changed(self):
        # Called when the user adjusts the min/max region handles
        self._store_current_state()

    def _store_current_state(self):
        # Make sure we have a dict for saving full states
        if not hasattr(self, 'view_states'):
            self.view_states = {}

        # Save the entire item state, e.g. { 'levels': [...], 'gradient': {...}, ... }
        state = self.saveState()
        self.view_states[self._view] = state

    def mousePressEvent(self, event):
        if not self.show_histogram:
            event.accept()  # Block interaction
        else:
            # Call the cached base-class handler explicitly.
            self._orig_mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not self.show_histogram:
            event.accept()
        else:
            self._orig_mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if not self.show_histogram:
            event.accept()
        else:
            self._orig_mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        # Make sure the region is visible
        self.region.setVisible(True)

        # Actually auto‐range the region
        self.autoHistogramRange()

        self.autoRange()
        # Force a redraw
        self.update()
        self._store_current_state()

        # Do NOT call the parent double-click handler:
        # pg.HistogramLUTItem.mouseDoubleClickEvent(self, event)

        # Accept the event to prevent it from bubbling up
        event.accept()
    
    def autoRange(self):
        data = self.imageitem.image
        if data is not None and data.size > 0:
            min_val, max_val = float(data.min()), float(data.max())
            self.setLevels(min_val, max_val)          # sets internal LUT mapping
            self.region.setRegion((min_val, max_val)) # sets the slider region


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

        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)
        
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
            
    # def paint(self, painter, option, widget=None):
        old_brush = self.gradient.gradRect.brush()      
        if getattr(self, 'discrete_mode', False):
            # In discrete mode, skip the default overlay drawing.
            # return
            self.gradient.gradRect.setBrush(QtGui.QBrush(QtCore.Qt.BrushStyle.NoBrush))
        # Otherwise, call the original painting.
        self._orig_paint(p, *args) # painter, option, widget)
        self.gradient.gradRect.setBrush(old_brush)
        self.gradient.backgroundRect.hide() # the cause of the hatching pattern 
        



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


from pyqtgraph import GraphicsObject
from PyQt6.QtWidgets import QGraphicsItem
from PyQt6.QtGui import QPainter
from PyQt6.QtCore import QRectF, Qt

import OpenGL.GL as gl

class GLPixelGridOverlay(GraphicsObject):
    """
    A QGraphicsItem that draws lines between centers of adjacent pixels
    (including diagonal neighbors) in an Nx x Ny grid, using raw OpenGL calls.
    """

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.reset()

    def _generate_grid_lines_8(self):
        """
        Build a (2*E, 2) float array of line endpoints for 8-neighbor adjacency
        (or however many directions are in self.parent.steps[:idx]).
        Also build self.lineIndices of shape (Ny, Nx, ndirs).

        This version does it in a vectorized manner to speed up initialization.
        """
        idx = self.parent.inds[0][0]
        neighbor_offsets = self.parent.steps[:idx]
        ndirs = len(neighbor_offsets)

        # Prepare container for line indices: lineIndices[j, i, d] = which line
        self.lineIndices = np.full((self.Ly, self.Lx, ndirs), -1, dtype=int)

        all_coords = []   # Will accumulate [ (x1, y1), (x2, y2), ... ]

        line_index = 0
        # Create a meshgrid for all pixel centers
        j_coords, i_coords = np.mgrid[0:self.Ly, 0:self.Lx]  # shape => (Ny, Nx)
        cx = i_coords + 0.5
        cy = j_coords + 0.5

        for d, (dy, dx) in enumerate(neighbor_offsets):
            # Compute neighbor coords
            ni = i_coords + dx
            nj = j_coords + dy

            # Mask in-bounds
            mask = (ni >= 0) & (ni < self.Lx) & (nj >= 0) & (nj < self.Ly)
            valid = np.where(mask)

            # For each valid pixel, we have a line from (cx, cy) to (nx, ny)
            x1 = cx[valid]
            y1 = cy[valid]
            x2 = x1 + dx
            y2 = y1 + dy

            n_valid = x1.size
            # Each line takes 2 vertices => 2*n_valid entries
            coords = np.empty((2*n_valid, 2), dtype=np.float32)

            coords[0::2, 0] = x1
            coords[0::2, 1] = y1
            coords[1::2, 0] = x2
            coords[1::2, 1] = y2

            # Set lineIndices for each valid pixel to the line_index range
            # each pixel gets exactly 1 line => line_index.. line_index + n_valid - 1
            # Note each line has two vertices, but line_index counts "lines" not "vertices"
            self.lineIndices[valid[0], valid[1], d] = np.arange(line_index, line_index + n_valid)

            all_coords.append(coords)
            line_index += n_valid

        # Concatenate everything
        self.num_lines = line_index
        vertex_data = np.concatenate(all_coords, axis=0)
        return vertex_data

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
        return QRectF(0, 0, self.Lx, self.Ly)

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
        alpha_scale = self.scale_to_alpha(scale, threshold=10.0, steepness=1/np.sqrt(2))
        
        # 4) Multiply the original alpha (self.base_alpha) by alpha_scale
        #    so lines that were hidden remain hidden (base_alpha=0).
        final_alpha = self.base_alpha * alpha_scale

        temp_color = self.color_data.copy()
        temp_color[:, 3] *= final_alpha
        # print(final_alpha[0],scale)

        # 6) Upload the color data to the GPU in one call
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.color_id)
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, temp_color.nbytes, temp_color)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

        # 7) Re-enter native painting for raw OpenGL calls
        painter.beginNativePainting()
        # painter.setCompositionMode(QtGui.QPainter.CompositionMode_Source)
        
        try:
            # Enable scissor if you want the viewbox clipping
            gl.glEnable(gl.GL_SCISSOR_TEST)
            gl.glEnable(gl.GL_MULTISAMPLE)

            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            # gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_SRC_ALPHA)
            # gl.glBlendFuncSeparate(gl.GL_ONE, gl.GL_ONE_MINUS_SRC_ALPHA, gl.GL_ONE, gl.GL_ONE_MINUS_SRC_ALPHA)

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
            # gl.glClearColor(0.0, 0.0, 0.0, 0.0)
            # gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

            gl.glDisable(gl.GL_MULTISAMPLE)
            gl.glDisable(gl.GL_SCISSOR_TEST)
            gl.glDisable(gl.GL_BLEND)
            

            painter.endNativePainting()  
    
    def toggle_lines_batch(self, indices, alpha=None, visible=False):
        """
        Hide or show all the lines in `indices` with a single GPU update.
        If visible=False, set alpha=0 for each line (i.e. line is invisible).
        If visible=True, set alpha=1 for each line (line is fully visible).
        
        Parameters
        ----------
        indices : list or array
            List of line indices to hide/show. Each line i corresponds to
            vertices [2*i, 2*i+2] in color_data.
        visible : bool
            If False, alpha=0 (invisible). If True, alpha=1 (visible).
        """
        new_alpha = 1.0 if visible else 0.0
        
        if alpha is None:
            alpha = [new_alpha]*len(indices)

        # 1) Modify self.color_data in CPU memory.
        for i,a in zip(indices, alpha):
            # Each line i has two vertices => color_data[2*i : 2*i+2].
            # RGBA => color_data[...] shape is (N,4).
            # self.color_data[2*i : 2*i+2, 3] = new_alpha
            
            # self.color_data[2*i : 2*i+2] = [a,a,a,a]
            self.color_data[2*i : 2*i+2] = a
            
        
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
        
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.color_id)
        gl.glBufferSubData(
            gl.GL_ARRAY_BUFFER,
            0, 
            self.color_data.nbytes,
            self.color_data
        )
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        self.update()
        
    def draw_all_lines(self):
        """Explicitly draw all lines based on the current color data."""
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.color_id)
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, self.color_data.nbytes, self.color_data)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        self.update()
    
    def scale_to_alpha(self, scale, threshold=1.0, steepness=4.0):
        """
        Returns a value in [0..1] that smoothly ramps up around 'threshold'.
        alpha = 1 / (1 + exp(-steepness*(scale - threshold)))
        """
        return 1.0 / (1.0 + np.exp(-steepness * (scale - threshold)))
        
    # def shape(self):
    #     """No mouse picking; return empty path."""
    #     return pg.QtGui.QPainterPath()

    def initialize_colors_from_affinity(self):
        """Initialize the colors using only the first 4 directions of the affinity graph."""
        affinity = self.parent.affinity_graph
        logger.info(f'inside initialize_colors_from_affinity {affinity.sum()}')
        
        if affinity is None:
            logger.info("Affinity graph is None")
            return

        # We only want to match the 4 directions in lineIndices, ignoring the symmetric ones
        affinity_4 = affinity[:4]  # shape => (4, Ny, Nx)

        # lineIndices is shape => (Ny, Nx, 4)
        ndirs_line = self.lineIndices.shape[2]
        ndirs_aff = affinity_4.shape[0]
        if ndirs_line != ndirs_aff:
            print(f"Mismatch: lineIndices has {ndirs_line} directions, but affinity[:4] has {ndirs_aff} directions.")
            return

        # Flatten lineIndices in (d, j, i) order => shape (4 * Ny * Nx,)
        lineIndices_t = self.lineIndices.transpose(2, 0, 1).ravel()

        # Flatten affinity_4 similarly => shape (4 * Ny * Nx,)
        affinity_t = affinity_4.reshape(ndirs_aff, -1).ravel()

        # Reset all colors (alpha=0)
        self.color_data[:] = 0.0

        # Find valid lines (where lineIndices >= 0)
        valid_mask = (lineIndices_t >= 0)
        valid_indices = lineIndices_t[valid_mask].astype(int)    # Actual line indices
        valid_alpha = affinity_t[valid_mask].astype(float)       # 1.0 or 0.0 from the affinity

        # Each line has 2 vertices => line i covers color_data[2*i : 2*i+2]
        pairs = np.stack([2 * valid_indices, 2 * valid_indices + 1], axis=1).ravel()
        # print('pairs', pairs.shape, valid_alpha.shape, self.color_data.shape, valid_indices.shape)
        # Assign alpha for both endpoints of each line
        # self.color_data[pairs] = np.repeat(valid_alpha, 2)
        # 1) Repeat alpha so both endpoints of each line get the same value
        vals1D = np.repeat(valid_alpha, 2)  # shape => (2 * num_valid_lines,)

        # 2) Expand to (2 * num_valid_lines, 1)
        vals2D = vals1D[:, None]

        # 3) Tile across 4 columns so R=G=B=A=alpha
        vals2D = np.tile(vals2D, (1, 4))  # shape => (2 * num_valid_lines, 4)

        # 4) Assign to those rows in self.color_data
        self.color_data[pairs] = vals2D

        # Upload to GPU
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.color_id)
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, self.color_data.nbytes, self.color_data)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

        self.update()
        
    def reset(self):
        """
        Re-initialize this overlay for a new grid size (Nx, Ny).
        Recompute vertex data, line indices, color buffers, etc.
        """
        # self.Lx = Nx
        # self.Ly = Ny

        # # 1) Regenerate the line coordinates
        # self.vertex_data = self._generate_grid_lines_8()  # uses updated Nx, Ny
        # # 2) Re-upload the vertex buffer
        # self.upload_data()

        # # 3) Reallocate color_data to match new self.num_lines
        # self.color_data = np.zeros((2*self.num_lines, 4), dtype=np.float32)
        # self.base_alpha = np.ones_like(self.color_data[:, 3])  # reset alpha=1

        # # 4) Re-initialize the color VBO
        # gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.color_id)
        # gl.glBufferData(gl.GL_ARRAY_BUFFER,
        #                 self.color_data.nbytes,
        #                 self.color_data,
        #                 gl.GL_DYNAMIC_DRAW)
        # gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

        # # 5) Optionally re-run 'initialize_colors_from_affinity' for the new shape
        # self.initialize_colors_from_affinity()
        # self.update()

                
        logger.info("inside reset of GLPixelGridOverlay")
        self.Lx = self.parent.Lx
        self.Ly = self.parent.Ly

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

        # Initialize colors based on the affinity graph
        self.initialize_colors_from_affinity()
        self.update()
        