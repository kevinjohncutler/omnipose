
from PyQt6 import QtGui, QtCore, QtWidgets

from ..guiparts import TOOLBAR_WIDTH, SPACING, WIDTH_0, TextField

INPUT_WIDTH = 2*WIDTH_0 + SPACING
WIDTH_3 = 3*WIDTH_0+2*SPACING
WIDTH_5 = 5*WIDTH_0+4*SPACING

from PyQt6 import QtGui, QtCore, QtWidgets
from PyQt6.QtCore import Qt, pyqtSlot, QCoreApplication
from PyQt6.QtWidgets import QMainWindow, QApplication, QWidget, QScrollBar, QComboBox, QGridLayout, QPushButton, QCheckBox, QLabel, QProgressBar, QLineEdit, QScrollArea
from PyQt6.QtGui import QPalette
import pyqtgraph as pg

import superqt

from .theme import COLORS
from .. import guiparts
import qtawesome as qta

from .. import logger, GAMMA_PATH, DEFAULT_MODEL

from cellpose_omni import models

def make_main_widget(self):
    # ---- MAIN WIDGET LAYOUT ---- #


    self.l0 = QGridLayout()
    self.scrollArea = QScrollArea()
    self.scrollArea.setStyleSheet('QScrollArea {border: none;}') # just for main window
    self.scrollArea.setWidgetResizable(True)
    # policy = QtWidgets.QSizePolicy()
    # policy.setRetainSizeWhenHidden(True)
    # self.scrollArea.setSizePolicy(policy)

    self.cwidget = QWidget(self)
    self.cwidget.setLayout(self.l0) 
    self.scrollArea.setWidget(self.cwidget)


    self.setCentralWidget(self.scrollArea)

    # s = int(SPACING/pxr)
    s = int(SPACING)
    self.l0.setVerticalSpacing(s)
    self.l0.setHorizontalSpacing(s)
    # self.l0.setRetainSizeWhenHidden(True)
    self.l0.setContentsMargins(10,10,10,10)


def dropdowns(self,width=WIDTH_0):
    return ''.join(['QComboBox QAbstractItemView {',
                    # 'background-color: #151515;',
                    # 'selection-color: white; ',
                    'min-width: {};'.format(width),
                    '}'])
    

def enable_buttons(self):
    if len(self.model_strings) > 0:
        # self.ModelButton.setStyleSheet(self.styleUnpressed)
        self.ModelButton.setEnabled(True)
    # CP2.0 buttons disabled for now     
    # self.StyleToModel.setStyleSheet(self.styleUnpressed)
    # self.StyleToModel.setEnabled(True)
    # for i in range(len(self.StyleButtons)):
    #     self.StyleButtons[i].setEnabled(True)
    #     self.StyleButtons[i].setStyleSheet(self.styleUnpressed)
    
    # self.SizeButton.setEnabled(True)
    # self.PencilCheckBox.setEnabled(True)
    # self.SizeButton.setStyleSheet(self.styleUnpressed)
    # self.newmodel.setEnabled(True)
    self.loadMasks.setEnabled(True)
    self.saveSet.setEnabled(True)
    self.savePNG.setEnabled(True)
    # self.saveServer.setEnabled(True)
    self.saveOutlines.setEnabled(True)
    self.toggle_mask_ops()
    
    self.threshslider.setEnabled(True)
    self.probslider.setEnabled(True)

    self.update_plot()
    self.setWindowTitle(self.filename)

def make_buttons(self):
    label_style = ''
    COLORS[0] = '#545454'
    self.boldfont = QtGui.QFont("Arial")
    self.boldfont.setPixelSize(18)
    self.boldfont.setWeight(QtGui.QFont.Weight.Bold)

    self.boldfont_button = QtGui.QFont("Arial")
    self.boldfont_button.setPixelSize(14)
    self.boldfont_button.setWeight(QtGui.QFont.Weight.Bold)
    
    self.medfont = QtGui.QFont("Arial")
    self.medfont.setPixelSize(13)
    self.smallfont = QtGui.QFont("Arial")
    self.smallfont.setPixelSize(12)

    self.checkstyle = ''
    self.linestyle = ''

    label = QLabel('Views:')#[\u2191 \u2193]')
    label.setStyleSheet('color: {}'.format(COLORS[0]))
    label.setFont(self.boldfont)
    self.l0.addWidget(label, 0,0,1,-1)

    
    b=1
    c = TOOLBAR_WIDTH//2 
    
    
    self.view = 0 # 0=image, 1=flowsXY, 2=flowsZ, 3=cellprob
    self.color = 0 # 0=RGB, 1=gray, 2=R, 3=G, 4=B
    self.ViewChoose = guiparts.ViewRadioButtons(self, row=b, col=0)

    
    b+=5
    # c+=1
    label = QLabel('color map:')
    label.setToolTip('Built-in pyqtgraph color maps')
    label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    label.setStyleSheet(label_style)
    label.setFont(self.medfont)
    self.l0.addWidget(label, b, c-1,1,2)

            
    # One way of ordering the cmaps I want and including all the rest
    builtin = pg.graphicsItems.GradientEditorItem.Gradients.keys()
    self.default_cmaps = ['grey','cyclic','magma','viridis']
    self.cmaps = self.default_cmaps+list(set(builtin) - set(self.default_cmaps))
    self.RGBDropDown = QComboBox()
    self.RGBDropDown.addItems(self.cmaps)
    self.RGBDropDown.setFont(self.smallfont)
    self.RGBDropDown.currentIndexChanged.connect(lambda: self.color_choose())
    self.RGBDropDown.setFixedWidth(WIDTH_3)
    self.RGBDropDown.setStyleSheet(self.dropdowns(width=WIDTH_3))
    
    c+=1
    self.l0.addWidget(self.RGBDropDown, b, c,1, TOOLBAR_WIDTH-c)
            
    
    b = 1
    self.quadrant_label = QLabel('sectors')
    self.quadrant_label.setFixedWidth(WIDTH_3)

    self.quadrant_label.setToolTip('Double-click anywhere in the image to re-center')
    self.quadrant_label.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
    
    self.quadrant_label.setStyleSheet(label_style)
    self.quadrant_label.setFont(self.boldfont_button)
    

    c = TOOLBAR_WIDTH//2 
    self.l0.addWidget(self.quadrant_label, b,TOOLBAR_WIDTH-3,1,3)
    guiparts.make_quadrants(self, b+1)
    
    
    # cross-hair
    self.vLine = pg.InfiniteLine(angle=90, movable=False)
    self.hLine = pg.InfiniteLine(angle=0, movable=False)
    self.vLineOrtho = [pg.InfiniteLine(angle=90, movable=False), pg.InfiniteLine(angle=90, movable=False)]
    self.hLineOrtho = [pg.InfiniteLine(angle=0, movable=False), pg.InfiniteLine(angle=0, movable=False)]

    # b-=1
    c0 = TOOLBAR_WIDTH//2 - 1
    self.orthobtn = QCheckBox('orthoviews')
    # self.orthobtn.setLayoutDirection(Qt.RightToLeft)
    self.orthobtn.setStyleSheet(label_style)
    self.orthobtn.setStyleSheet("margin-left:50%;")
    self.orthobtn.setToolTip('activate orthoviews with 3D image')
    self.orthobtn.setFont(self.boldfont_button)
    self.orthobtn.setChecked(False)
    self.l0.addWidget(self.orthobtn, b,c0,1,2)
    self.orthobtn.toggled.connect(lambda: self.toggle_ortho())
    
    b+=1

    # add z position 
    self.currentZ = 0
    label = QLabel('Z slice:')
    label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    label.setStyleSheet(label_style)
    label.setFont(self.medfont)

    self.l0.addWidget(label, b,c0-1,1,2)
    self.zpos = QLineEdit()
    # self.zpos.setStyleSheet(self.textbox_style)
    self.zpos.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter | QtCore.Qt.AlignmentFlag.AlignVCenter)
    # self.zpos.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
    self.zpos.setText(str(self.currentZ))
    # self.zpos.returnPressed.connect(lambda: self.compute_scale())
    self.zpos.setFixedWidth(INPUT_WIDTH)
    self.zpos.setFixedHeight(WIDTH_0)
    
    # self.l0.addWidget(self.zpos, b, c,1, TOOLBAR_WIDTH-c)
    self.l0.addWidget(self.zpos, b, c0+1,1, 1)
    

    b+=1
    label = QLabel('slice width:')
    label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    label.setStyleSheet(label_style)
    label.setFont(self.medfont)
    self.l0.addWidget(label, b,c0-1,1,2)
    self.dz = 11
    self.dzedit = QLineEdit()
    # self.dzedit.setStyleSheet(self.textbox_style)
    self.dzedit.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
    self.dzedit.setText(str(self.dz))
    self.dzedit.returnPressed.connect(lambda: self.update_ortho())
    self.dzedit.setFixedWidth(INPUT_WIDTH)
    self.dzedit.setFixedHeight(WIDTH_0)
    
    # self.l0.addWidget(self.dzedit, b, c,1, TOOLBAR_WIDTH-c)
    self.l0.addWidget(self.dzedit, b, c0+1,1, 1)
    # c-w,1, w)

    b+=1
    label = QLabel('aspect ratio:')
    label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    label.setStyleSheet(label_style)
    label.setFont(self.medfont)
    self.l0.addWidget(label, b,c0-1,1,2)
    self.zaspect = 1.0
    self.zaspectedit = QLineEdit()
    # self.zaspectedit.setStyleSheet(self.textbox_style)
    self.zaspectedit.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
    self.zaspectedit.setText(str(self.zaspect))
    self.zaspectedit.returnPressed.connect(lambda: self.update_ortho())
    self.zaspectedit.setFixedWidth(INPUT_WIDTH)
    self.zaspectedit.setFixedHeight(WIDTH_0)
    
    self.l0.addWidget(self.zaspectedit, b, c0+1,1, 1)


    # b+=1
    # label = QLabel('Image rescaling/gamma:')
    # label.setStyleSheet('color: {}'.format(COLORS[0]))
    # label.setFont(self.boldfont)
    # self.l0.addWidget(label, b,0,1,8)
    
    
    # turn this into enterable upper and lower bounds 
    #b+=1
    #self.autochannelbtn = QCheckBox('renormalize channels')
    #self.autochannelbtn.setStyleSheet(self.checkstyle)
    #self.autochannelbtn.setFont(self.medfont)
    #self.autochannelbtn.setChecked(True)
    #self.autochannelbtn.setToolTip('sets channels so that 1st and 99th percentiles at same values, only works for 2D images currently')
    #self.l0.addWidget(self.autochannelbtn, b,0,1,4)

    
    # b+=1
    # label = QLabel('percentile clipping:')
    # label.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
    # label.setFont(self.medfont)
    # self.l0.addWidget(label, b,1,1,TOOLBAR_WIDTH-1)
    
    # used in io to rescale, need to aslo adda toggle to use this to override Omnipose rescaling (or rather, input the rescaled image vs raw)
    
    b+=4
    self.autobtn = QCheckBox('auto-adjust')
    self.autobtn.setStyleSheet(self.checkstyle)
    self.autobtn.setFont(self.medfont)
    self.autobtn.setChecked(True)
    self.l0.addWidget(self.autobtn, b,0,1,1)
    b+=1
    
    # use inverted image for running cellpose
    self.invert = QCheckBox('invert image')
    self.invert.setStyleSheet(self.checkstyle)
    self.invert.setFont(self.medfont)
    self.l0.addWidget(self.invert, b,0,1,1)
    self.invert.toggled.connect(lambda: self.update_plot())
    

    # IMAGE RESCALING slider
    b-=1
    c = 1
    self.contrast_slider = superqt.QLabeledDoubleRangeSlider(QtCore.Qt.Orientation.Horizontal)
    self.contrast_slider.setDecimals(2) 
    # self.contrast_slider.label_shift_x = -10
    
    self.contrast_slider.setHandleLabelPosition(superqt.QLabeledRangeSlider.LabelPosition.NoLabel)
    self.contrast_slider.setEdgeLabelMode(superqt.QLabeledRangeSlider.EdgeLabelMode.LabelIsValue)
    self.contrast_slider.valueChanged.connect(lambda: self.level_change())


    # CONSTRAST SLIDER
    self.contrast_slider.setMinimum(0.0)
    self.contrast_slider.setMaximum(100.0)
    self.contrast_slider.setValue((0.1,99.9))  
    self.contrast_slider._max_label.setFont(self.medfont)
    self.contrast_slider._min_label.setFont(self.medfont)
    self.l0.addWidget(self.contrast_slider, b,c+1,1,TOOLBAR_WIDTH-(c+1))

    
    icon = qta.icon("ph.scissors-fill", color="#888888")  # Use scissors icon with color
    label = QLabel()  # Create a QLabel to hold the icon
    label.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)

    # Set the QtAwesome icon as a pixmap
    font_size = self.boldfont.pixelSize()+2
    pixmap = icon.pixmap(font_size, font_size)  # Set size to match previous font size
    label.setPixmap(pixmap)
    label.setStyleSheet("margin-left: 50px;")  # Adjust '10px' to control the shift
    # Add the label to the layout
    self.l0.addWidget(label, b, c, 1, 1)
    
    # GAMMA SLIDER
    b+=1
    # button = QPushButton('')
    # button.setIcon(QtGui.QIcon(str(GAMMA_PATH)))
    # button.setStyleSheet("QPushButton {Text-align: middle; background-color: none;}")
    # button.setDefault(True)
    # self.l0.addWidget(button, b,c,1,1)
    
    # Create a QtAwesome icon
    # icon = qta.icon("mdi6.gamma", color="#888888")  
    # label = QLabel()
    # label.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
    # icon_size = 24  # Desired icon size
    # label.setPixmap(icon.pixmap(icon_size, icon_size))  # Set the pixmap with the specified size
    # label.setStyleSheet("margin-left: 50px;")  # Adjust '10px' to control the shift
    # self.l0.addWidget(label, b, c, 1, 1) # Add the label to the layout

    label = QLabel()
    icon = QtGui.QIcon(str(GAMMA_PATH))  # Use your gamma SVG
    pixmap = icon.pixmap(16, 16)  # Adjust size of the SVG rendering
    label.setPixmap(pixmap)
    label.setStyleSheet("margin-left: 50px;")  # Add left margin
    label.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
    self.l0.addWidget(label, b, c, 1, 1)

#  .py, .rst, 
    
    self.gamma = 1.0
    self.gamma_slider = superqt.QLabeledDoubleSlider(QtCore.Qt.Orientation.Horizontal)
    self.gamma_slider.valueChanged.connect(lambda: self.gamma_change())
    self.gamma_slider.setMinimum(0)
    self.gamma_slider.setMaximum(2)
    self.gamma_slider.setValue(self.gamma) 
    self.gamma_slider._label.setFont(self.medfont)
    self.l0.addWidget(self.gamma_slider, b,c+1,1,TOOLBAR_WIDTH-(c+1))
    
    
    
    # b+=1
    # self.l0.addWidget(QLabel(''),b,0,2,4)
    # self.l0.setRowStretch(b, 1)


    # self.resize = -1
    self.X2 = 0

    # b+=1
    # line = QHLine()
    # line.setStyleSheet(self.linestyle)
    # self.l0.addWidget(line, b,0,1,TOOLBAR_WIDTH)
    b += 2
    label = QLabel('Drawing:')
    label.setStyleSheet('color: {}'.format(COLORS[0]))
    label.setFont(self.boldfont)
    self.l0.addWidget(label, b, 0, 1, TOOLBAR_WIDTH)
    
    c = TOOLBAR_WIDTH // 2
    

    # Combine pen size and active checkbox into one row
    b += 1
    # label = QLabel('pen:')
    # label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    # label.setStyleSheet(label_style)
    # label.setFont(self.medfont)
    # self.l0.addWidget(label, b, 0, 1, 1)

    # Pen Active Checkbox
    # self.PencilCheckBox = QCheckBox()
    # self.PencilCheckBox.setStyleSheet(checkstyle(COLORS[0]))
    # self.PencilCheckBox.setFont(self.medfont)
    # self.PencilCheckBox.setChecked(True)  # Default pen active
    # self.PencilCheckBox.toggled.connect(lambda: self.autosave_on())
    # self.PencilCheckBox.toggled.connect(lambda: self.update_brush_slider_color())
    # self.l0.addWidget(self.PencilCheckBox, b, 1, 1, 1)
    
    ### PENCIL 
# Define the active and inactive icons

    # Create the IconToggleButton
    self.PencilCheckBox = guiparts.IconToggleButton(
        icon_active_name="mdi.draw",  # Active icon name
        icon_inactive_name="mdi.draw",  # Active icon name
        
        # icon_inactive_name="mdi6.pencil-minus",  # Inactive icon name
        # icon_inactive_name="mdi6.pencil-minus",  # Inactive icon name
        inactive_color="#888",  # Custom inactive color
        icon_size=36,
        parent=self
    )
    
    self.PencilCheckBox.setEnabled(True)  
    self.PencilCheckBox.setChecked(False) 
    
    
    self.PencilCheckBox.toggled.connect(lambda: self.autosave_on())
    self.PencilCheckBox.toggled.connect(lambda: self.draw_change())

    # Add the button to your layout
    self.l0.addWidget(self.PencilCheckBox, b, c, 1, 1)
    

    
    # pencil width slider
    self.brush_size = 1
    self.brush_slider = superqt.QLabeledSlider(QtCore.Qt.Orientation.Horizontal)
    self.brush_slider.setMinimum(1)
    self.brush_slider.setMaximum(9) # could make this bigger, but people should not make crude masks 
    self.brush_slider.setSingleStep(2)
    self.brush_slider.setValue(self.brush_size) 
    self.brush_slider.valueChanged.connect(lambda: self.brush_size_change())


    # Set the text color based on self.PencilCheckBox state
    # self.update_brush_slider_color()


    # Add the slider to the layout
    # self.l0.addWidget(self.pencil_label, b, c, 1, 1)  # Pencil icon on the left
    self.l0.addWidget(self.brush_slider, b, c+1, 1, TOOLBAR_WIDTH-(c+1))


    # Active Label Input
    b += 1
    self.current_label = 0  # Default active label
    label = QLabel('active label:')
    label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    label.setStyleSheet(label_style)
    label.setFont(self.medfont)
    text = (
        'You can manually enter the active label for drawing.\n'
        'Press the “P” key and click on the image to pick a label from the view.'
    )
    label.setToolTip(text)

    self.LabelInput = QLineEdit()
    self.LabelInput.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
    self.LabelInput.setToolTip(text)
    self.LabelInput.setText(str(self.current_label))  # Default to 0
    self.LabelInput.setFont(self.medfont)
    self.LabelInput.returnPressed.connect(lambda: self.update_active_label())

    # Add Label to Layout
    c = TOOLBAR_WIDTH // 2
    self.l0.addWidget(label, b, c - 1, 1, 3)  # Match `nchan` label position

    # Add Input Field
    c += 1
    self.LabelInput.setFixedWidth(INPUT_WIDTH)  # Ensure consistent width
    self.LabelInput.setFixedHeight(WIDTH_0)    # Ensure consistent height
    self.l0.addWidget(self.LabelInput, b, TOOLBAR_WIDTH - 2, 1, 2)  # Match `nchan` input field position

    # MASK TOGGLE
    b -= 1
    self.layer_off = False
    self.masksOn = True
    self.MCheckBox = QCheckBox('masks')
    self.MCheckBox.setToolTip('Press M to toggle masks')
    self.MCheckBox.setStyleSheet(self.checkstyle)
    self.MCheckBox.setFont(self.medfont)
    self.MCheckBox.setChecked(self.masksOn)
    self.MCheckBox.toggled.connect(lambda: self.toggle_masks())
    self.l0.addWidget(self.MCheckBox, b, 0, 1, 2)

    # NCOLOR TOGGLE
    b += 1
    self.ncolor = True
    self.NCCheckBox = QCheckBox('n-color')
    self.NCCheckBox.setToolTip('Press C or N to toggle n-color masks')
    self.NCCheckBox.setStyleSheet(self.checkstyle)
    self.NCCheckBox.setFont(self.medfont)
    self.NCCheckBox.setChecked(self.ncolor)
    self.NCCheckBox.toggled.connect(lambda: self.toggle_ncolor())
    self.l0.addWidget(self.NCCheckBox, b, 0, 1, 2)

    # OUTLINE TOGGLE
    b += 1
    self.outlinesOn = True  # Turn on by default
    self.OCheckBox = QCheckBox('outlines')
    self.OCheckBox.setToolTip('Press Z or O to toggle outlines')
    self.OCheckBox.setStyleSheet(self.checkstyle)
    self.OCheckBox.setFont(self.medfont)
    self.OCheckBox.setChecked(False)
    self.OCheckBox.toggled.connect(lambda: self.toggle_masks())
    self.l0.addWidget(self.OCheckBox, b, 0, 1, 2)
    
    # AFFINITY GRAPH TOGGLE
    b += 1
    self.agridOn = False  # Turn off by default
    self.ACheckBox = QCheckBox('affinity graph')
    self.ACheckBox.setToolTip('Press A toggle affinity graph overlay')
    self.ACheckBox.setStyleSheet(self.checkstyle)
    self.ACheckBox.setFont(self.medfont)
    self.ACheckBox.setChecked(False)
    # self.ACheckBox.toggled.connect(lambda: self.toggle_affiniy_graph())
    # self.ACheckBox.toggled.connect(lambda x: print("Checkbox toggled:", x))
    # self.ACheckBox.toggled.connect(lambda: self.affinityOverlay.toggle())
    self.ACheckBox.toggled.connect(lambda x: self.pixelGridOverlay.setVisible(x))
    
    # self.ACheckBox.toggled.connect(self.toggle_affinity)
    
    self.l0.addWidget(self.ACheckBox, b, 0, 1, 2)
    
    

    # CROSSHAIR TOGGLE
    # b -= 1  # Adjust layout row
    c = TOOLBAR_WIDTH // 2 # reset column position
    self.CHCheckBox = QCheckBox('cross-hairs')
    self.CHCheckBox.setStyleSheet(self.checkstyle)
    self.CHCheckBox.setFont(self.medfont)
    self.CHCheckBox.toggled.connect(lambda: self.cross_hairs())
    self.l0.addWidget(self.CHCheckBox, b, c, 1, TOOLBAR_WIDTH)

#### The segmentation section is where a lot of rearrangement happened 
    
    b+=2
    label = QLabel('Segmentation:')
    label.setStyleSheet('color: {}'.format(COLORS[0]))
    label.setFont(self.boldfont)
    self.l0.addWidget(label, b,0,1,TOOLBAR_WIDTH)

    # I got rid of calibrate diameter, manual is far faster

    #DIMENSION text field 
    b+=1
    c = TOOLBAR_WIDTH//2
    label = QLabel('dimension:')
    label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    label.setStyleSheet(label_style)
    label.setFont(self.medfont)
    self.l0.addWidget(label, b, c-1, 1,2)
    c+=1
    self.dim = 2
    self.Dimension = QComboBox()
    self.Dimension.addItems(["2","3","4"])
    # self.Dimension.currentIndexChanged.connect(lambda: self.brush_choose())
    self.Dimension.setStyleSheet(self.dropdowns())
    self.Dimension.setFont(self.medfont)
    self.Dimension.setFixedWidth(WIDTH_3)
    self.l0.addWidget(self.Dimension, b, c,1, TOOLBAR_WIDTH-c)
    
    # CELL DIAMETER text field
    b+=1
    self.diameter = 0
    label = QLabel('cell diameter:')
    label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    label.setStyleSheet(label_style)
    label.setFont(self.medfont)
    text = ('you can manually enter the approximate diameter for your cells, '
                '\nor press “calibrate” to let the model estimate it. '
                '\nThe size is represented by a disk at the bottom of the view window '
                '\n(can turn this disk off by unchecking “scale disk visible”)')
    label.setToolTip(text)
    self.Diameter = QLineEdit()
    self.Diameter.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
    self.Diameter.setToolTip(text)
    self.Diameter.setText(str(self.diameter))
    self.Diameter.setFont(self.medfont)
    # self.Diameter.returnPressed.connect(lambda: self.compute_scale())
    c = TOOLBAR_WIDTH//2
    # self.l0.addWidget(label, b, c, 1,2)
    self.l0.addWidget(label, b, c-1, 1,3)

    c+=1
    self.Diameter.setFixedWidth(INPUT_WIDTH) # twice width plus spacing
    self.Diameter.setFixedHeight(WIDTH_0)
    self.l0.addWidget(self.Diameter, b, TOOLBAR_WIDTH-2,1,2)
            

    # NCHAN text field
    b+=1
    self.nchan = 1
    label = QLabel('nchan:')
    label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    label.setStyleSheet(label_style)
    label.setFont(self.medfont)
    text = ('Custom models need the number of image channels to be specified. '
                '\nFor newer Omnipose models: the actual number of channels of your images.  '
                '\nFor Cellpose or older Omnipose: always 2. '
                )
    label.setToolTip(text)
    self.ChanNumber = QLineEdit()
    self.ChanNumber.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
    self.ChanNumber.setToolTip(text)
    self.ChanNumber.setText(str(self.nchan))
    self.ChanNumber.setFont(self.medfont)
    self.ChanNumber.returnPressed.connect(lambda: self.set_nchan())        
    c = TOOLBAR_WIDTH//2
    # self.l0.addWidget(label, b, c, 1,2)
    self.l0.addWidget(label, b, c-1, 1,3)

    c+=1
    self.ChanNumber.setFixedWidth(INPUT_WIDTH) # twice width plus spacing
    self.ChanNumber.setFixedHeight(WIDTH_0)
    self.l0.addWidget(self.ChanNumber, b, TOOLBAR_WIDTH-2,1,2)

    # AFFINITY seg 
    b-=2
    self.AffinityCheck = QCheckBox('affinity graph reconstruction')
    self.AffinityCheck.setStyleSheet(self.checkstyle)
    self.AffinityCheck.setFont(self.medfont)
    self.AffinityCheck.setChecked(False)
    self.AffinityCheck.setToolTip('sets whether or not to use affinity graph mask reconstruction')
    self.AffinityCheck.toggled.connect(lambda: self.toggle_affinity())
    self.l0.addWidget(self.AffinityCheck, b,0,1,2)

    # SCALE DISK toggle
    # b+=1
    # self.scale_on = True
    # self.ScaleOn = QCheckBox('scale disk visible')
    # self.ScaleOn.setStyleSheet(self.checkstyle)
    # self.ScaleOn.setFont(self.medfont)
    # # self.ScaleOn.setStyleSheet('color: red;')
    # self.ScaleOn.setChecked(self.scale_on)
    # self.ScaleOn.setToolTip('see current diameter as red disk at bottom')
    # self.ScaleOn.toggled.connect(lambda: self.toggle_scale())
    # # self.toggle_scale() # toggle (off) 
    # self.l0.addWidget(self.ScaleOn, b,0,1,2)
    # # self.toggle_scale() # toggle (off) 
    
    # SIZE MODEL
    # b+=1
    # self.SizeModel = QCheckBox('SizeModel rescaling')
    # self.SizeModel.setStyleSheet(self.checkstyle)
    # self.SizeModel.setFont(self.medfont)
    # self.SizeModel.setChecked(False)
    # self.SizeModel.setToolTip('sets whether or not to use a SizeModel for rescaling \nprior to running network')
    # self.l0.addWidget(self.SizeModel, b,0,1,2)

    # TILING CHECKBOX
    b+=1
    self.tile = QCheckBox('tile for inference')
    self.tile.setStyleSheet(self.checkstyle)
    self.tile.setFont(self.medfont)
    self.tile.setChecked(False)
    self.tile.setToolTip('sets whether or not to split the image into tiles for inference')
    self.l0.addWidget(self.tile, b,0,1,2)
    


    # BOUNDARY FIELD CHECKBOX (sets nclasses to 3 instead of 2)
    # This should be done automatically by the model in the final version
    b+=1
    self.boundary = QCheckBox('boundary field output')
    self.boundary.setStyleSheet(self.checkstyle)
    self.boundary.setFont(self.medfont)
    self.boundary.setChecked(False)
    self.boundary.setToolTip('sets whether or not the model was trained with a boundary field \n(older Omnpose models)')
    self.l0.addWidget(self.boundary, b,0,1,2)
    

    # Disabling this for now, always do fast
    b+=1
    self.NetAvg = QComboBox()
    self.NetAvg.setStyleSheet(self.dropdowns(width=WIDTH_5))
    self.NetAvg.setFixedWidth(WIDTH_5)
    self.NetAvg.addItems(['average 4 nets', 'run 1 net', '1 net + no resample'])
    self.NetAvg.setFont(self.smallfont)
    self.NetAvg.setToolTip('average 4 different fit networks (default); run 1 network (faster); or run 1 net + turn off resample (fast)')
    self.l0.addWidget(self.NetAvg, b,TOOLBAR_WIDTH//2,1,TOOLBAR_WIDTH-TOOLBAR_WIDTH//2)

    # MODEL DROPDOWN
    b+=1
    self.ModelChoose = QComboBox()
    if len(self.model_strings) > len(models.MODEL_NAMES):
        current_index = len(models.MODEL_NAMES)
        self.NetAvg.setCurrentIndex(1)
    else:
        current_index = models.MODEL_NAMES.index(DEFAULT_MODEL)
    self.ModelChoose.addItems(self.model_strings) #added omnipose model names
    self.ModelChoose.setStyleSheet(self.dropdowns(width=WIDTH_5))
    self.ModelChoose.setFont(self.smallfont)
    self.ModelChoose.setCurrentIndex(current_index)
    self.ModelChoose.activated.connect(lambda: self.model_choose())
    # self.ModelChoose.activated.connect(lambda idx: self.model_choose(idx))
    
    
    self.l0.addWidget(self.ModelChoose, b, TOOLBAR_WIDTH//2,1,TOOLBAR_WIDTH-TOOLBAR_WIDTH//2)


    label = QLabel('model:')
    label.setStyleSheet(label_style)
    label.setFont(self.medfont)
    #update tooltip string 
    tipstr = ('there is a <em>cyto</em> model,'
                'a new <em>cyto2</em> model from user submissions,'
                'a <em>nuclei</em> model,'
                'and several Cellpose and Omnipose models for:'
                'phase contrast bacteria (<em>bact_phase_cp/_omni</em>)'
                'fluorescent bacteria (<em>bact_fluor_cp/_omni</em>)'
                'membrane-tagged 3D A. thaliana (<em>plant_cp/_omni</em>)'
                'brighfield C. elegans ((<em>worm_cp/_omni</em>))'
                'brighfield C. elegans and phase contrast bacteria((<em>worm_bact_omni</em>))'
                'high-res C. elegans ((<em>worm_high_res_omni</em>))')
    label.setToolTip(tipstr)
    self.ModelChoose.setToolTip(tipstr)
    self.ModelChoose.setFixedWidth(WIDTH_5)
    self.l0.addWidget(label, b, 0,1,TOOLBAR_WIDTH//2)





    # CHANNELS DROPDOWN
    b+=1
    self.ChannelChoose = [QComboBox(), QComboBox()]
    self.ChannelChoose[0].addItems(['gray','red','green','blue'])
    self.ChannelChoose[1].addItems(['none','red','green','blue'])
    cstr = ['chan to segment:', 'chan2 (optional):']
    for i in range(2):
        self.ChannelChoose[i].setFixedWidth(WIDTH_5)
        self.ChannelChoose[i].setStyleSheet(self.dropdowns(width=WIDTH_5))
        self.ChannelChoose[i].setFont(self.smallfont)
        label = QLabel(cstr[i])
        label.setStyleSheet(label_style)
        label.setFont(self.medfont)
        if i==0:
            label.setToolTip('this is the channel in which the cytoplasm or nuclei exist that you want to segment')
            self.ChannelChoose[i].setToolTip('this is the channel in which the cytoplasm or nuclei exist that you want to segment')
        else:
            label.setToolTip(('if <em>cytoplasm</em> model is chosen, and you also have a nuclear channel, '
                                'then choose the nuclear channel for this option'))
            self.ChannelChoose[i].setToolTip(('if <em>cytoplasm</em> model is chosen, and you also have a nuclear channel, '
                                                'then choose the nuclear channel for this option'))
        self.l0.addWidget(label, b, 0,1,TOOLBAR_WIDTH-TOOLBAR_WIDTH//2)
        self.l0.addWidget(self.ChannelChoose[i], b, TOOLBAR_WIDTH//2,1,TOOLBAR_WIDTH-TOOLBAR_WIDTH//2)

        b+=1
    



    # THRESHOLDS
    b+=1
    c = TOOLBAR_WIDTH//2 

    label = QLabel('mask:')
    label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    
    label.setToolTip('threshold on scalar output field to seed cell masks \n'
                    '(set lower to include more pixels)')
    label.setStyleSheet(label_style)
    label.setFont(self.medfont)
    self.l0.addWidget(label, b, c-1,1,1)
    
    self.cellprob = 0.0
    self.probslider = superqt.QLabeledDoubleSlider(QtCore.Qt.Horizontal)
    self.probslider._label.setFont(self.smallfont)
    self.probslider.setRange(-6,6)
    self.probslider.setValue(self.cellprob)
    self.probslider._label.setFont(self.medfont)


    # self.probslider.setFont(self.medfont)
    # font = QtGui.QFont("Arial")
    # font.setPixelSize(1)   
    # self.probslider.setFont(font)
    self.l0.addWidget(self.probslider, b, c,1,TOOLBAR_WIDTH-c)
    self.probslider.setEnabled(False)
    self.probslider.valueChanged.connect(lambda: self.run_mask_reconstruction())
    

    b+=1
    label = QLabel('flow:')
    label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    
    label.setToolTip(('threshold on flow match to accept a mask \n(Set higher to remove poor predictions.'
                        'Diabled with 0 by default.)'))
    label.setStyleSheet(label_style)
    label.setFont(self.medfont)
    self.l0.addWidget(label, b, c-1,1,1)
    
    # b+=1
    self.threshold = 0.0
    self.threshslider = superqt.QLabeledDoubleSlider(QtCore.Qt.Horizontal)
    self.threshslider.setRange(0,1)
    self.threshslider.setValue(self.threshold)
    self.threshslider._label.setFont(self.medfont)
    self.l0.addWidget(self.threshslider, b, c,1,TOOLBAR_WIDTH-c)
    self.threshslider.setEnabled(False)
    self.threshslider.valueChanged.connect(lambda: self.run_mask_reconstruction())
    

    
    
    # use GPU
    b-=1
    self.useGPU = QCheckBox('use GPU')
    self.useGPU.setStyleSheet(self.checkstyle)
    self.useGPU.setFont(self.medfont)
    self.useGPU.setToolTip(('if you have specially installed the <i>cuda</i> version of mxnet, '
                            'then you can activate this, but it won’t give huge speedups when running single 2D images in the GUI.'))
    self.check_gpu()
    self.l0.addWidget(self.useGPU, b,0,1,1)
    
    
    # verbose
    b+=1
    self.verbose = QCheckBox('verbose')
    # self.verbose.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignTop)
    self.verbose.setStyleSheet(self.checkstyle)
    self.verbose.setFont(self.medfont)
    self.verbose.setChecked(False)
    self.verbose.setToolTip('sets whether or not to output verbose text to terminal for debugging')
    self.l0.addWidget(self.verbose, b,0,1,1)
    
    # show the exact parameters used 
    # b-=1
    b+=2
    

    c = 1
    label = QLabel('Click to copy segmentation parameters:')
    label.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
    label.setStyleSheet(label_style)
    label.setFont(self.medfont)
    self.l0.addWidget(label, b,c,1,TOOLBAR_WIDTH-c)
    
    
    b+=1
    h = 95
    self.runstring = TextField(self)
    self.runstring.setFixedHeight(h)
    self.runstring.setFont(self.medfont)
    self.l0.addWidget(self.runstring, b,c,1,TOOLBAR_WIDTH-c)
    self.runstring.clicked.connect(lambda: self.copy_runstring())

    self.ModelButton = QPushButton('Segment\nimage')
    
    self.ModelButton.clicked.connect(lambda: self.compute_model())
    self.l0.addWidget(self.ModelButton, b,0,1,c)
    self.ModelButton.setEnabled(False)

    self.ModelButton.setFont(self.boldfont_button)
    self.ModelButton.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
    self.ModelButton.setFixedHeight(h)
    self.ModelButton.setFixedWidth(h)


    
    
    b+=1


    self.roi_count = QLabel('0 RoIs')
    # self.roi_count.setStyleSheet('color: white;')
    self.roi_count.setFont(self.boldfont_button)
    self.roi_count.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
    w = TOOLBAR_WIDTH//2+1
    self.l0.addWidget(self.roi_count, b, w,1,TOOLBAR_WIDTH-w)
    
    
    self.progress = QProgressBar(self)
    self.progress.setValue(0)
    self.l0.addWidget(self.progress, b,0,1,w)
    

    # add scrollbar underneath
    self.scroll = QScrollBar(QtCore.Qt.Horizontal)
    # self.scroll.setMaximum(10)
    self.scroll.valueChanged.connect(lambda: self.move_in_Z())
    self.l0.addWidget(self.scroll, b,TOOLBAR_WIDTH+1,1,3*b)
    
    # self.l0.addWidget(QLabel(''), b, 0,1,TOOLBAR_WIDTH)        


    self.toggleArrow = QtWidgets.QToolButton(self)
    self.toggleArrow.setFixedSize(24, 24)
    self.toggleArrow.setArrowType(QtCore.Qt.ArrowType.LeftArrow)
    self.toggleArrow.setToolTip("Toggle Links Editor")
    self.toggleArrow.clicked.connect(self.toggleLinksDock)

    # Place it in the same row (b) but in a column to the right.
    # For example, if you want it at the far right, you can
    # add it to the same columns as the scroll bar but with right alignment:
    self.l0.addWidget(
        self.toggleArrow,
        b,
        TOOLBAR_WIDTH + 3,      # pick a column beyond your normal controls
        1,
        1,
        alignment=Qt.AlignmentFlag.AlignRight
    )



    # ---- drawing area ---- #
    self.win = pg.GraphicsLayoutWidget()
    self.win.viewport().setAttribute(QtCore.Qt.WidgetAttribute.WA_AcceptTouchEvents, False)
    # self.win.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    # self.win.setAutoFillBackground(False)
                
    
    self.l0.addWidget(self.win, 0, TOOLBAR_WIDTH+1, b, 3*b)
    self.win.scene().sigMouseClicked.connect(lambda event: self.plot_clicked(event))
    self.win.scene().sigMouseMoved.connect(lambda pos: self.mouse_moved(pos))
    self.make_viewbox()
    self.make_orthoviews()
    self.l0.setColumnStretch(TOOLBAR_WIDTH+1, 1)
    # self.l0.setMaximumWidth(100)
    # self.ScaleOn.setChecked(False)  # can only toggle off after make_viewbox is called 

    print('buttons',b)