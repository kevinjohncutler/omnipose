import signal, sys, os, pathlib, warnings, datetime, time

# def handle_exception(exc_type, exc_value, exc_traceback):
#     if issubclass(exc_type, KeyboardInterrupt):
#         sys.__excepthook__(exc_type, exc_value, exc_traceback)
#         return
#     print("Uncaught exception:", exc_type, exc_value)
#     import traceback
#     traceback.print_tb(exc_traceback)

# sys.excepthook = handle_exception

# os.environ["QT_DEBUG_PLUGINS"] = "1"

import numpy as np
# np.seterr(all='raise')  # Raise exceptions instead of warnings


from PyQt6 import QtGui, QtCore, QtWidgets
from PyQt6.QtCore import Qt, pyqtSlot, QCoreApplication
from PyQt6.QtWidgets import QMainWindow, QApplication, QWidget, QScrollBar, QComboBox, QGridLayout, QPushButton, QCheckBox, QLabel, QProgressBar, QLineEdit, QScrollArea
from PyQt6.QtGui import QPalette
import pyqtgraph as pg

# for cursor
from PyQt6.QtWidgets import QGraphicsPathItem
from PyQt6.QtGui import QPen, QBrush, QPainterPath, QTransform
from PyQt6.QtGui import QCursor
from PyQt6.QtCore import QPointF


from pyqtgraph import ViewBox

os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '1'

from scipy.stats import mode
# from scipy.ndimage import gaussian_filter

from . import guiparts, menus, io
from .. import models, dynamics
from ..utils import download_url_to_file, masks_to_outlines, diameters 
from ..io import get_image_files, imsave, imread, check_dir #OMNI_INSTALLED
from ..transforms import resize_image #fixed import
from ..plot import disk
from omnipose.utils import normalize99, to_8_bit
from omnipose import core, gpu, utils, misc

OMNI_INSTALLED = 1
from .guiutils import checkstyle, get_unique_points, avg3d, interpZ

logger = io.logger

from .guiparts import TOOLBAR_WIDTH, SPACING, WIDTH_0, TextField, QHLine

ALLOWED_THEMES = ['light','dark']

INPUT_WIDTH = 2*WIDTH_0 + SPACING
WIDTH_3 = 3*WIDTH_0+2*SPACING
WIDTH_5 = 5*WIDTH_0+4*SPACING

import darkdetect
import qdarktheme
import superqt
import qtawesome as qta


# no more matplotlib just for colormaps
from cmap import Colormap



#logo 
ICON_PATH = pathlib.Path.home().joinpath('.omnipose','logo.png')
ICON_URL = 'https://github.com/kevinjohncutler/omnipose/blob/main/gui/logo.png?raw=true'


#test files
op_dir = pathlib.Path.home().joinpath('.omnipose','test_files')
check_dir(op_dir)
files = ['Sample000033.png','Sample000193.png','Sample000252.png','Sample000306.tiff','e1t1_crop.tif']
test_images = [pathlib.Path.home().joinpath(op_dir, f) for f in files]
for path,file in zip(test_images,files):
    if not path.is_file():
        download_url_to_file('https://github.com/kevinjohncutler/omnipose/blob/main/docs/test_files/'+file+'?raw=true',
                                path, progress=True)
PRELOAD_IMAGE = str(test_images[-1])
DEFAULT_MODEL = 'bact_phase_omni'


from omnipose.utils import sinebow
# from colour import rgb2hex
from matplotlib.colors import rgb2hex

N = 29
c = sinebow(N)
COLORS = [rgb2hex(c[i][:3]) for i in range(1,N+1)] #can only do RBG, not RGBA for stylesheet


if not ICON_PATH.is_file():
    print('downloading logo from', ICON_URL,'to', ICON_PATH)
    download_url_to_file(ICON_URL, ICON_PATH, progress=True)

# Not everyone with have a math font installed, so all this effort just to have
# a cute little math-style gamma as a slider label...
GAMMA_PATH = pathlib.Path.home().joinpath('.omnipose','gamma.svg')
BRUSH_PATH = pathlib.Path.home().joinpath('.omnipose','brush.svg')

GAMMA_URL = 'https://github.com/kevinjohncutler/omnipose/blob/main/gui/gamma.svg?raw=true'   
if not GAMMA_PATH.is_file():
    print('downloading gamma icon from', GAMMA_URL,'to', GAMMA_PATH)
    download_url_to_file(GAMMA_URL, GAMMA_PATH, progress=True)
    

def run(image=PRELOAD_IMAGE):
    start_time = time.time()  # Record start time
    
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # Always start by initializing Qt (only once per application)
    warnings.filterwarnings("ignore")
    QCoreApplication.setApplicationName('Omnipose')
    app = QApplication(sys.argv)

    screen = app.primaryScreen()
    dpi = screen.logicalDotsPerInch()
    pxr = screen.devicePixelRatio()
    size = screen.availableGeometry()
    clipboard = app.clipboard()

    qdarktheme.clear_cache()

    icon_path = pathlib.Path.home().joinpath('.cellpose', 'logo.png')
    guip_path = pathlib.Path.home().joinpath('.cellpose', 'cellpose_gui.png')
    style_path = pathlib.Path.home().joinpath('.cellpose', 'style_choice.npy')
    if not icon_path.is_file():
        cp_dir = pathlib.Path.home().joinpath('.cellpose')
        cp_dir.mkdir(exist_ok=True)
        logger.info('downloading logo')
        download_url_to_file('https://www.cellpose.org/static/images/cellpose_transparent.png', icon_path, progress=True)
    if not guip_path.is_file():
        logger.info('downloading help window image')
        download_url_to_file('https://www.cellpose.org/static/images/cellpose_gui.png', guip_path, progress=True)
    if not style_path.is_file():
        logger.info('downloading style classifier')
        download_url_to_file('https://www.cellpose.org/static/models/style_choice.npy', style_path, progress=True)
    
    app_icon = QtGui.QIcon()
    icon_path = str(ICON_PATH.resolve())
    for i in [16,24,32,48,64,256]:
        app_icon.addFile(icon_path, QtCore.QSize(i,i)) 
    app.setWindowIcon(app_icon) 
    
    # models.download_model_weights() # does not exist
    win = MainW(size, dpi, pxr, clipboard, image=image)

    # the below code block will automatically toggle the theme with the system,
    # but the manual color definitions (everywhere I set a style sheet) can mess that up
    @pyqtSlot()
    def sync_theme_with_system() -> None:
        theme = str(darkdetect.theme()).lower()
        theme = theme if theme in ALLOWED_THEMES else 'dark' #default to dark theme 
        stylesheet = qdarktheme.load_stylesheet(theme)
        QApplication.instance().setStyleSheet(stylesheet)
        win.darkmode = theme=='dark'
        win.accent = win.palette().brush(QPalette.ColorRole.Highlight).color()
        if hasattr(win,'win'):
            win.win.setBackground("k" if win.darkmode else '#f0f0f0') #pull out real colors from theme here from example
       
       # explicitly set colors for items that don't change automatically with theme
        win.set_hist_colors()
        win.set_button_color()
        win.set_crosshair_colors()
        win.SCheckBox.update_icons() 
        # win.update_plot()
    app.paletteChanged.connect(sync_theme_with_system)             
    sync_theme_with_system()

    end_time = time.time()  # Record end time
    print(f"Total Time: {end_time - start_time:.4f} seconds")


    ret = app.exec()
    sys.exit(ret)
    

class MainW(QMainWindow):
    def __init__(self, size, dpi, pxr, clipboard, image=None):
        start_time = time.time()  # Record start time

        super(MainW, self).__init__()
        # palette = app.palette()
        # palette.setColor(QPalette.ColorRole.ColorRole.Link, dark_palette.link().color())
        # app.setPalette(palette)

        # print(qdarktheme.load_palette().link().color())
        self.darkmode = str(darkdetect.theme()).lower() in ['none','dark'] # have to initialize; str catches None on some systems

        pg.setConfigOptions(imageAxisOrder="row-major")
        self.clipboard = clipboard
        # geometry that works on mac and ubuntu at least 
        Y = int(925 - (25*dpi*pxr)/24)
        self.setGeometry(100, 100, min(1200,size.width()),  min(Y,size.height())) 

        # self.showMaximized()
        self.setWindowTitle("Omnipose GUI")
        self.cp_path = os.path.dirname(os.path.realpath(__file__))

        menus.mainmenu(self)
        menus.editmenu(self)
        menus.modelmenu(self)
        # menus.helpmenu(self) # all of these are outdated 
        menus.omnimenu(self)

        self.model_strings = models.MODEL_NAMES.copy()
        self.loaded = False

        # ---- MAIN WIDGET LAYOUT ---- #

        scrollable = 1 
        if scrollable:
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

            self.scrollArea.setMinimumSize(self.cwidget.sizeHint())

            self.setCentralWidget(self.scrollArea)
        else:
            self.cwidget = QWidget(self)
            self.l0 = QGridLayout()
            self.cwidget.setLayout(self.l0)
            self.setCentralWidget(self.cwidget)
        
        # s = int(SPACING/pxr)
        s = int(SPACING)
        self.l0.setVerticalSpacing(s)
        self.l0.setHorizontalSpacing(s)
        # self.l0.setRetainSizeWhenHidden(True)
        self.l0.setContentsMargins(10,10,10,10)
        

        self.imask = 0

        b = self.make_buttons()
        # self.cwidget.setStyleSheet('border: 1px; border-radius: 10px')
        
        
        # ---- drawing area ---- #
        self.win = pg.GraphicsLayoutWidget()
        self.win.viewport().setAttribute(QtCore.Qt.WidgetAttribute.WA_AcceptTouchEvents, False)
        self.l0.addWidget(self.win, 0, TOOLBAR_WIDTH+1, b, 3*b)
        self.win.scene().sigMouseClicked.connect(self.plot_clicked)
        self.win.scene().sigMouseMoved.connect(self.mouse_moved)
        self.make_viewbox()
        self.make_orthoviews()
        self.l0.setColumnStretch(TOOLBAR_WIDTH+1, 1)
        # self.l0.setMaximumWidth(100)
        self.ScaleOn.setChecked(False)  # can only toggle off after make_viewbox is called 

        # hard-coded colormaps entirely replaced with pyqtgraph

        # Instantiate the Colormap object
        cmap = Colormap("gist_ncar")

        # Generate 1,000,000 evenly spaced color samples
        colormap = cmap(np.linspace(0, 1, 1000000))  # Directly call the colormap
        colormap = (np.array(colormap) * 255).astype(np.uint8)  # Convert to uint8

        # Stable random shuffling of colors
        np.random.seed(42)
        self.colormap = colormap[np.random.permutation(1000000)]
    
        self.undo_stack = []  # Stack to store cellpix history
        self.redo_stack = []  # Stack to store redo states
        self.max_undo_steps = 50  # Limit the number of undo steps
    

        self.is_stack = True # always loading images of same FOV? Not sure about this assumption...
        # if called with image, load it
        if image is not None:
            self.filename = image
            print('loading', self.filename)
            io._load_image(self, self.filename)

        # training settings
        d = datetime.datetime.now()
        self.training_params = {'model_index': 0,
                                'learning_rate': 0.1, 
                                'weight_decay': 0.0001, 
                                'n_epochs': 100,
                                'model_name':'CP' + d.strftime("_%Y%m%d_%H%M%S")
                               }
        


        self.setAcceptDrops(True)

        self.win.show()
        self.show()
        
        end_time = time.time()  # Record end time
        print(f"Init Time: {end_time - start_time:.4f} seconds")
        
    def save_state(self):
        """Save the current state of cellpix for undo."""
        if len(self.undo_stack) >= self.max_undo_steps:
            self.undo_stack.pop(0)
        self.undo_stack.append(np.copy(self.cellpix))

    def undo_action(self):
        """Undo the last action."""
        if self.undo_stack:
            # Save the current state for redo
            self.redo_stack.append(np.copy(self.cellpix))
            if len(self.redo_stack) >= self.max_undo_steps:
                self.redo_stack.pop(0)  # Limit redo stack size

            # Restore the last state from the undo stack
            self.cellpix = self.undo_stack.pop()        
            self.update_layer()  # Refresh the display

        else:
            print("Nothing to undo.")
            
    def redo_action(self):
        """Redo the last undone action."""
        if self.redo_stack:

            # Save the current state for undo
            self.undo_stack.append(np.copy(self.cellpix))
            if len(self.undo_stack) >= self.max_undo_steps:
                self.undo_stack.pop(0)  # Limit undo stack size
                
            # Restore the last state from the redo stack
            self.cellpix = self.redo_stack.pop()
            self.update_layer()  # Refresh the display
        

        else:
            print("Nothing to redo.")
    
    def update_layer(self):
        logger.info(f'updating layer {self.loaded}')
        self.draw_layer()
        self.update_roi_count()
        self.win.show()
        self.show()
        
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
        self.RGBChoose = guiparts.RGBRadioButtons(self, row=b, col=0)

        
        
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
        self.RGBDropDown.currentIndexChanged.connect(self.color_choose)
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
        self.orthobtn.toggled.connect(self.toggle_ortho)
        
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
        self.zpos.returnPressed.connect(self.compute_scale)
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
        self.dzedit.returnPressed.connect(self.update_ortho)
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
        self.zaspectedit.returnPressed.connect(self.update_ortho)
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
        self.invert.toggled.connect(self.update_plot)
        

        # IMAGE RESCALING slider
        b-=1
        c = 1
        self.slider = superqt.QLabeledDoubleRangeSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setDecimals(2) 
        # self.slider.label_shift_x = -10
        
        self.slider.setHandleLabelPosition(superqt.QLabeledRangeSlider.LabelPosition.NoLabel)
        self.slider.setEdgeLabelMode(superqt.QLabeledRangeSlider.EdgeLabelMode.LabelIsValue)
        self.slider.valueChanged.connect(self.level_change)

        # self.slider.setStyleSheet(guiparts.horizontal_slider_style())

        # CONSTRAST SLIDER
        self.slider.setMinimum(0.0)
        self.slider.setMaximum(100.0)
        self.slider.setValue((0.1,99.9))  
        self.slider._max_label.setFont(self.medfont)
        self.slider._min_label.setFont(self.medfont)
        self.l0.addWidget(self.slider, b,c+1,1,TOOLBAR_WIDTH-(c+1))

        
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


        
        self.gamma = 1.0
        self.gamma_slider = superqt.QLabeledDoubleSlider(QtCore.Qt.Orientation.Horizontal)
        # self.gamma_slider.setHandleLabelPosition(superqt.QLabeledSlider.LabelPosition.NoLabel)
        # self.gamma_slider.setEdgeLabelMode(superqt.QLabeledSlider.EdgeLabelMode.LabelIsValue)
        self.gamma_slider.valueChanged.connect(self.gamma_change)

        # self.gamma_slider.setStyleSheet(guiparts.horizontal_slider_style())
        
        self.gamma_slider.setMinimum(0)
        self.gamma_slider.setMaximum(2)
        self.gamma_slider.setValue(self.gamma) 
        self.gamma_slider._label.setFont(self.medfont)
        self.l0.addWidget(self.gamma_slider, b,c+1,1,TOOLBAR_WIDTH-(c+1))
        
        
        
        # b+=1
        # self.l0.addWidget(QLabel(''),b,0,2,4)
        # self.l0.setRowStretch(b, 1)


        self.resize = -1
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
        # self.SCheckBox = QCheckBox()
        # self.SCheckBox.setStyleSheet(checkstyle(COLORS[0]))
        # self.SCheckBox.setFont(self.medfont)
        # self.SCheckBox.setChecked(True)  # Default pen active
        # self.SCheckBox.toggled.connect(self.autosave_on)
        # self.SCheckBox.toggled.connect(self.update_brush_slider_color)
        # self.l0.addWidget(self.SCheckBox, b, 1, 1, 1)
        
      ### PENCIL 
    # Define the active and inactive icons

        # Create the IconToggleButton
        self.SCheckBox = guiparts.IconToggleButton(
            icon_active_name="mdi.draw",  # Active icon name
            icon_inactive_name="mdi.draw",  # Active icon name
            
            # icon_inactive_name="mdi6.pencil-minus",  # Inactive icon name
            # icon_inactive_name="mdi6.pencil-minus",  # Inactive icon name
            inactive_color="#888",  # Custom inactive color
            icon_size=36,
            parent=self
        )
        
        self.SCheckBox.setChecked(True)  # Default pen active
        self.SCheckBox.setEnabled(True)  # Default pen active
        
        self.SCheckBox.toggled.connect(self.autosave_on)
        self.SCheckBox.toggled.connect(self.draw_change)

        # Add the button to your layout
        self.l0.addWidget(self.SCheckBox, b, c, 1, 1)
        

        
        # pencil width slider
        self.brush_size = 1
        self.brush_slider = superqt.QLabeledSlider(QtCore.Qt.Orientation.Horizontal)
        self.brush_slider.setMinimum(1)
        self.brush_slider.setMaximum(9) # could make this bigger, but people should not make crude masks 
        self.brush_slider.setSingleStep(2)
        self.brush_slider.setValue(self.brush_size) 
        self.brush_slider.valueChanged.connect(self.brush_size_change)


        # Set the text color based on self.SCheckBox state
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
        self.LabelInput.returnPressed.connect(self.update_active_label)

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
        self.MCheckBox.toggled.connect(self.toggle_masks)
        self.l0.addWidget(self.MCheckBox, b, 0, 1, 2)

        # NCOLOR TOGGLE
        b += 1
        self.ncolor = True
        self.NCCheckBox = QCheckBox('n-color')
        self.NCCheckBox.setToolTip('Press C or N to toggle n-color masks')
        self.NCCheckBox.setStyleSheet(self.checkstyle)
        self.NCCheckBox.setFont(self.medfont)
        self.NCCheckBox.setChecked(self.ncolor)
        self.NCCheckBox.toggled.connect(self.toggle_ncolor)
        self.l0.addWidget(self.NCCheckBox, b, 0, 1, 2)

        # OUTLINE TOGGLE
        b += 1
        self.outlinesOn = False  # Turn off by default
        self.OCheckBox = QCheckBox('outlines')
        self.OCheckBox.setToolTip('Press Z or O to toggle outlines')
        self.OCheckBox.setStyleSheet(self.checkstyle)
        self.OCheckBox.setFont(self.medfont)
        self.OCheckBox.setChecked(False)
        self.OCheckBox.toggled.connect(self.toggle_masks)
        self.l0.addWidget(self.OCheckBox, b, 0, 1, 2)

        # CROSSHAIR TOGGLE
        # b -= 1  # Adjust layout row
        c = TOOLBAR_WIDTH // 2 # reset column position
        self.CHCheckBox = QCheckBox('cross-hairs')
        self.CHCheckBox.setStyleSheet(self.checkstyle)
        self.CHCheckBox.setFont(self.medfont)
        self.CHCheckBox.toggled.connect(self.cross_hairs)
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
        # self.Dimension.currentIndexChanged.connect(self.brush_choose)
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
        self.Diameter.returnPressed.connect(self.compute_scale)
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
        self.ChanNumber.returnPressed.connect(self.set_nchan)        
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
        self.AffinityCheck.toggled.connect(self.toggle_affinity)
        self.l0.addWidget(self.AffinityCheck, b,0,1,2)

        # SCALE DISK toggle
        b+=1
        self.scale_on = True
        self.ScaleOn = QCheckBox('scale disk visible')
        self.ScaleOn.setStyleSheet(self.checkstyle)
        self.ScaleOn.setFont(self.medfont)
        # self.ScaleOn.setStyleSheet('color: red;')
        self.ScaleOn.setChecked(self.scale_on)
        self.ScaleOn.setToolTip('see current diameter as red disk at bottom')
        self.ScaleOn.toggled.connect(self.toggle_scale)
        # self.toggle_scale() # toggle (off) 
        self.l0.addWidget(self.ScaleOn, b,0,1,2)
        # self.toggle_scale() # toggle (off) 
        
        # SIZE MODEL
        b+=1
        self.SizeModel = QCheckBox('SizeModel rescaling')
        self.SizeModel.setStyleSheet(self.checkstyle)
        self.SizeModel.setFont(self.medfont)
        self.SizeModel.setChecked(False)
        self.SizeModel.setToolTip('sets whether or not to use a SizeModel for rescaling \nprior to running network')
        self.l0.addWidget(self.SizeModel, b,0,1,2)


        # BOUNDARY FIELD CHECKBOX (sets nclasses to 3 instead of 2)
        b+=1
        self.boundary = QCheckBox('boundary field output')
        self.boundary.setStyleSheet(self.checkstyle)
        self.boundary.setFont(self.medfont)
        self.boundary.setChecked(False)
        self.boundary.setToolTip('sets whether or not the model was trained with a boundary field \n(older Omnpose models)')
        self.l0.addWidget(self.boundary, b,0,1,2)
        

        # Disabling this for now, always do fast
        # b+=1
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
        self.ModelChoose.activated.connect(self.model_choose)
        
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
        self.probslider.valueChanged.connect(self.run_mask_reconstruction)
        

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
        self.threshslider.valueChanged.connect(self.run_mask_reconstruction)
        

        
        
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
        self.runstring.clicked.connect(self.copy_runstring)

        self.ModelButton = QPushButton('Segment \nimage')
        
        self.ModelButton.clicked.connect(self.compute_model)
        self.l0.addWidget(self.ModelButton, b,0,1,c)
        self.ModelButton.setEnabled(False)

        self.ModelButton.setFont(self.boldfont_button)
        self.ModelButton.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        self.ModelButton.setFixedHeight(h)
        self.ModelButton.setFixedWidth(h)


        
        
        b+=1


        self.roi_count = QLabel('0 ROIs')
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
        self.scroll.valueChanged.connect(self.move_in_Z)
        self.l0.addWidget(self.scroll, b,TOOLBAR_WIDTH+1,1,3*b)
        
        # self.l0.addWidget(QLabel(''), b, 0,1,TOOLBAR_WIDTH)        

        
        return b
        
    
        
    def update_brush_slider_color(self):
        """Update brush slider text color based on pen active state."""
        color = 'red' if self.SCheckBox.isChecked() else "gray"
        # print('color',self.accent)
        self.brush_slider.setStyleSheet(f"color: {color};")
    
    
    def brush_size_change(self):
        """Update the brush size based on slider value."""
        if self.loaded:
            value = self.brush_slider.value()
            
            # make sure this is odd
            odd_value = value | 1  # Ensure the value is odd by setting the least significant bit
            if odd_value != value:  # Only update if the value changes
                self.brush_slider.setValue(odd_value)
                value = odd_value
            
            self.ops_plot = {'brush_size': value}
            self.brush_size = value
            
            self.layer._generateKernel(self.brush_size)
            self.compute_kernel_path(self.layer._kernel)
            self.update_highlight()
            
    def draw_change(self):
        if not self.SCheckBox.isChecked():
            self.highlight_rect.hide()
        else:
            self.update_highlight()
            
                    
    def plot_clicked(self, event):
        if event.button()==QtCore.Qt.LeftButton and (event.modifiers() != QtCore.Qt.ShiftModifier and
                    event.modifiers() != QtCore.Qt.AltModifier):
            if event.double():
                self.recenter()
            elif self.loaded and not self.in_stroke:
                if self.orthobtn.isChecked():
                    items = self.win.scene().items(event.scenePos())
                    for x in items:
                        if x==self.p0:
                            pos = self.p0.mapSceneToView(event.scenePos())
                            x = int(pos.x())
                            y = int(pos.y())
                            if y>=0 and y<self.Ly and x>=0 and x<self.Lx:
                                self.yortho = y 
                                self.xortho = x
                                self.update_ortho()

    
    def dropdowns(self,width=WIDTH_0):
        return ''.join(['QComboBox QAbstractItemView {',
                        # 'background-color: #151515;',
                        # 'selection-color: white; ',
                        'min-width: {};'.format(width),
                        '}'])
        
    def update_active_label(self):
        """Update self.current_label from the input field."""
        try:
            self.current_label = int(self.LabelInput.text())
            print(f"Active label updated to: {self.current_label}")
        except ValueError:
            print("Invalid label input.")
            
    def update_active_label_field(self):
        """Sync the active label input field with the current label."""
        self.LabelInput.setText(str(self.current_label))
        
    def keyReleaseEvent(self, event):
    
        # drag / pan
        if event.key() == QtCore.Qt.Key_Space:
            self.spacePressed = False
        
        # pick and fill 
        elif event.key() == QtCore.Qt.Key_G:
            self.flood_fill_enabled = False
        elif event.key() == QtCore.Qt.Key_P:
            self.pick_label_enabled = False
        
        super().keyReleaseEvent(event)


    def keyPressEvent(self, event):
        if not self.loaded:
            return  # Do nothing if not loaded

        modifiers = event.modifiers()
        key = event.key()
    
        # Modifier-based actions (e.g., Undo/Redo)
        if modifiers & QtCore.Qt.ControlModifier:
            if key == QtCore.Qt.Key_Z:  # Ctrl+Z: Undo
                if modifiers & QtCore.Qt.ShiftModifier:  # Ctrl+Shift+Z: Redo
                    self.redo_action()
                else:
                    self.undo_action()

        # Actions based on individual keys
        elif key == QtCore.Qt.Key_Space:
            self.spacePressed = True  # Enable pan mode
        elif key == QtCore.Qt.Key_G:
            self.flood_fill_enabled = True  # Enable flood fill
        elif key == QtCore.Qt.Key_B:
            self.SCheckBox.toggle()  # Toggle brush tool
        elif key == QtCore.Qt.Key_M:
            self.MCheckBox.toggle()  # Toggle masks
        elif key == QtCore.Qt.Key_O:
            self.OCheckBox.toggle()  # Toggle outlines
        elif key == QtCore.Qt.Key_C or key == QtCore.Qt.Key_N:
            self.NCCheckBox.toggle()  # Toggle ncolor
        elif key == QtCore.Qt.Key_H:
            self.CHCheckBox.toggle()  # Toggle crosshairs

        # Navigation keys (Z-stack navigation)
        elif key == QtCore.Qt.Key_A:
            if self.NZ == 1:
                self.get_prev_image()
            else:
                self.currentZ = max(0, self.currentZ - 1)
                self.scroll.setValue(self.currentZ)
        elif key == QtCore.Qt.Key_D:
            if self.NZ == 1:
                self.get_next_image()
            else:
                self.currentZ = min(self.NZ - 1, self.currentZ + 1)
                self.scroll.wsetValue(self.currentZ)

        # Color cycling
        elif key == QtCore.Qt.Key_W:
            self.color = (self.color - 1) % len(self.cmaps)  # Cycle backward
            self.RGBDropDown.setCurrentIndex(self.color)
        elif key == QtCore.Qt.Key_S:
            self.color = (self.color + 1) % len(self.cmaps)  # Cycle forward
            self.RGBDropDown.setCurrentIndex(self.color)
        elif key == QtCore.Qt.Key_R:
            self.color = 1 if self.color != 1 else 0  # Toggle between 0 and 1
            self.RGBDropDown.setCurrentIndex(self.color)

        # Brush size adjustment
        elif key in {QtCore.Qt.Key_BracketLeft, QtCore.Qt.Key_BracketRight}:
            current_value = self.brush_slider.value()
            
            if key == QtCore.Qt.Key_BracketLeft:
                # Attempt to decrease the brush size
                if current_value > self.brush_slider.minimum():
                    new_value = current_value - self.brush_slider.singleStep()
                else:
                    # If already at the minimum, turn off drawing
                    new_value = self.brush_slider.minimum()
                    self.SCheckBox.setChecked(False)
            else:  # Key_BracketRight
                # Attempt to increase the brush size
                if not self.SCheckBox.isChecked():
                    # If the checkbox is unchecked, reset to minimum value and enable
                    new_value = self.brush_slider.minimum()
                    self.SCheckBox.setChecked(True)
                else:
                    new_value = min(self.brush_slider.maximum(), current_value + self.brush_slider.singleStep())

            self.brush_slider.setValue(new_value)  # Update the slider directly

    
        # Active label selection / color picking 
        elif event.key() in range(Qt.Key_0, Qt.Key_9 + 1):  # Numeric keys
            self.current_label = event.key() - Qt.Key_0  # Map keys to numbers 0-9
            self.update_active_label_field() # Sync with input field
            
            print(f"Active label set to: {self.current_label}")
            
        elif key == QtCore.Qt.Key_P or key == QtCore.Qt.Key_I:
            self.pick_label_enabled = True  # Enable label picking
                
        super().keyPressEvent(event)

    def check_gpu(self, use_torch=True):
        # also decide whether or not to use torch
        self.torch = use_torch
        self.useGPU.setChecked(False)
        self.useGPU.setEnabled(False)    
        if self.torch and gpu.use_gpu(use_torch=True)[-1]:
            self.useGPU.setEnabled(True)
            self.useGPU.setChecked(True)
        else:
            self.useGPU.setStyleSheet("color: rgb(80,80,80);")

    def get_channels(self):
        channels = [self.ChannelChoose[0].currentIndex(), self.ChannelChoose[1].currentIndex()]
        if self.current_model=='nuclei':
            channels[1] = 0

        if self.nchan==1:
            channels = None
        return channels

    def model_choose(self, index):
        if index > 0:
            logger.info(f'selected model {self.ModelChoose.currentText()}, loading now')
            self.initialize_model()
            # self.diameter = self.model.diam_labels
            
            # only set this when selected, not if user chooses a new value 
            bacterial = 'bact' in self.current_model
            if bacterial:
                self.diameter = 0.
                self.Diameter.setText('%0.1f'%self.diameter)
            else:
                self.diameter = float(self.Diameter.text())
            
            
            logger.info(f'diameter set to {self.diameter: 0.2f} (but can be changed)')

    # two important things: invert size added, and initialize model takes care of selecting a model

    def calibrate_size(self):
        self.initialize_model()
        diams, _ = self.model.sz.eval(self.stack[self.currentZ].copy(), invert=self.invert.isChecked(),
                                   channels=self.get_channels(), progress=self.progress)
        diams = np.maximum(5.0, diams)
        logger.info('estimated diameter of cells using %s model = %0.1f pixels'%
                (self.current_model, diams))
        self.Diameter.setText('%0.1f'%diams)
        self.diameter = diams
        self.compute_scale()
        self.progress.setValue(100)

    def toggle_scale(self):
        if self.scale_on:
            self.p0.removeItem(self.scale)
            self.scale_on = False
        else:
            self.p0.addItem(self.scale)
            self.scale_on = True
        self.recenter()    

    def toggle_affinity(self):
        self.recomoute_masks = True
        self.run_mask_reconstruction()
        
    def toggle_removals(self):
        if self.ncells>0:
            self.ClearButton.setEnabled(True)
            # self.remcell.setEnabled(True)
            self.undo.setEnabled(True)
        else:
            self.ClearButton.setEnabled(False)
            # self.remcell.setEnabled(False)
            self.undo.setEnabled(False)

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

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if os.path.splitext(files[0])[-1] == '.npy':
            io._load_seg(self, filename=files[0])
        else:
            io._load_image(self, filename=files[0])

    def toggle_masks(self):
        if self.MCheckBox.isChecked():
            self.masksOn = True
            self.restore_masks = True
        else:
            self.masksOn = False
            self.restore_masks = False
            
        if self.OCheckBox.isChecked():
            self.outlinesOn = True
        else:
            self.outlinesOn = False
        if not self.masksOn and not self.outlinesOn:
            self.p0.removeItem(self.layer)
            self.layer_off = True
        else:
            if self.layer_off:
                self.p0.addItem(self.layer)
            self.draw_layer()
            self.update_layer()
        if self.loaded:
            # self.update_plot()
            self.update_layer()

    def toggle_ncolor(self):
        if self.NCCheckBox.isChecked():
            self.ncolor = True
        else:
            self.ncolor = False
        io._masks_to_gui(self, format_labels=True)
        self.draw_layer()
        if self.loaded:
            # self.update_plot()
            self.update_layer()

    def level_change(self):
        if self.loaded:
            vals = self.slider.value()
            self.ops_plot = {'saturation': vals}
            self.saturation[self.currentZ] = vals
            self.update_plot()

    def move_in_Z(self):
        if self.loaded:
            self.currentZ = min(self.NZ, max(0, int(self.scroll.value())))
            self.zpos.setText(str(self.currentZ))
            self.update_plot()
            self.draw_layer()
            self.update_layer()
    
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
            invertY=True
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
        
        self.scale = pg.ImageItem(viewbox=self.p0, parent=self,levels=(0,255))
        
        self.p0.scene().contextMenuItem = self.p0
        self.p0.addItem(self.img)
        self.p0.addItem(self.layer)
        self.p0.addItem(self.scale)
                

        # Create the highlight path item
        self.highlight_rect = QGraphicsPathItem()
        # self.highlight_rect.setPen(Qt.PenStyle.NoPen) # no outline
        self.highlight_rect.setPen(QPen(Qt.PenStyle.NoPen))

        self.highlight_rect.setBrush(QBrush(pg.mkColor(255, 0, 0, 50)))  # Semi-transparent fill
        self.p0.addItem(self.highlight_rect)

        # Add the rectangle to the ViewBox, which aligns it with the image coordinates
        self.p0.addItem(self.highlight_rect)

        # Connect mouse movement signal
        self.p0.scene().sigMouseMoved.connect(self.update_highlight)
        
    
    def update_highlight(self, pos=None):
        if self.SCheckBox.isChecked():
            if pos is None:
                # Get the current global mouse position
                mouse_pos = QCursor.pos()  # Get the cursor position in global coordinates
                scene_mouse_pos = self.p0.scene().views()[0].mapFromGlobal(mouse_pos)  # Map to scene coordinates

                # Convert scene_mouse_pos to QPointF
                pos = QPointF(scene_mouse_pos.x(), scene_mouse_pos.y())

            # Map the cursor position from scene to view coordinates
            view_pos = self.p0.mapSceneToView(pos)

            # Get cursor position in image data coordinates
            x, y = int(view_pos.x()), int(view_pos.y())

            # Ensure the position is within the image bounds
            if x < 0 or y < 0 or x >= self.img.image.shape[1] or y >= self.img.image.shape[0]:
                self.highlight_rect.hide()
                return

            self.highlight_rect.show()
            
            # Get the kernel and its dimensions
            px = np.ones((1, 1))
            kernel = getattr(self.layer, '_kernel', px) if self.SCheckBox.isChecked() else px
            
            # Recompute the path if needed
            if not hasattr(self, 'highlight_path'):
                self.compute_kernel_path(kernel)

            # Position the cached path relative to the cursor
            transform = QTransform()
            transform.translate(x - kernel.shape[1] // 2, y - kernel.shape[0] // 2)
            transformed_path = transform.map(self.highlight_path)  # Apply transformation
            self.highlight_rect.setPath(transformed_path)

            


    def compute_kernel_path(self, kernel):
        path = QPainterPath()
        for ky, kx in np.argwhere(kernel == 1):
            path.addRect(kx, ky, 1, 1)  # Create the kernel path
        self.highlight_path = path

    def make_orthoviews(self):
        self.pOrtho, self.imgOrtho, self.layerOrtho = [], [], []
        for j in range(2):
            self.pOrtho.append(pg.ViewBox(
                                lockAspect=True,
                                name=f'plotOrtho{j}',
                                # border=[100, 100, 100],
                                invertY=True,
                                enableMouse=False
                            ))
            self.pOrtho[j].setMenuEnabled(False)

            self.imgOrtho.append(pg.ImageItem(viewbox=self.pOrtho[j], parent=self, levels=(0,255)))
            self.imgOrtho[j].autoDownsample = False

            self.layerOrtho.append(pg.ImageItem(viewbox=self.pOrtho[j], parent=self))
            self.layerOrtho[j].setLevels([0,255])

            #self.pOrtho[j].scene().contextMenuItem = self.pOrtho[j]
            self.pOrtho[j].addItem(self.imgOrtho[j])
            self.pOrtho[j].addItem(self.layerOrtho[j])
            self.pOrtho[j].addItem(self.vLineOrtho[j], ignoreBounds=False)
            self.pOrtho[j].addItem(self.hLineOrtho[j], ignoreBounds=False)
        
        self.pOrtho[0].linkView(self.pOrtho[0].YAxis, self.p0)
        self.pOrtho[1].linkView(self.pOrtho[1].XAxis, self.p0)
        
    def set_hist_colors(self):
        region = self.hist.region
        # c = self.palette().brush(QPalette.ColorRole.Text).color() # selects white or black from palette
        # selecting from the palette can be handy, but the corresponding colors in light and dark mode do not match up well
        color = '#44444450' if self.darkmode else '#cccccc50'
        # c.setAlpha(20)
        region.setBrush(color) # I hate the blue background
        
        c = self.accent
        c.setAlpha(60)
        region.setHoverBrush(c) # also the blue hover
        c.setAlpha(255) # reset accent alpha 
        
        color = '#777' if self.darkmode else '#aaa'
        pen =  pg.mkPen(color=color,width=1.5)
        ph =  pg.mkPen(self.accent,width=2)
        # region.lines[0].setPen(None)
        # region.lines[0].setHoverPen(color='c',width = 5)
        # region.lines[1].setPen('r')
        
        # self.hist.paint(self.hist.plot)
        # print('sss',self.hist.regions.__dict__)
        
        for line in region.lines:
            # c.setAlpha(100)
            line.setPen(pen)
            # c.setAlpha(200)
            line.setHoverPen(ph)
        
        self.hist.gradient.gradRect.setPen(pen)
        # c.setAlpha(100)
        self.hist.gradient.tickPen = pen
        self.set_tick_hover_color() 
        
        ax = self.hist.axis
        ax.setPen(color=(0,)*4) # transparent 
        # ax.setTicks([0,255])
        # ax.setStyle(stopAxisAtTick=(True,True))

        # self.hist = self.img.getHistogram()
        # self.hist.disableAutoHistogramRange()
        # c = self.palette().brush(QPalette.ColorRole.ToolTipBase).color() # selects white or black from palette
        # print(c.getRgb(),'ccc')
        
        # c.setAlpha(100)
        self.hist.fillHistogram(fill=True, level=1.0, color= '#222' if self.darkmode else '#bbb')
        self.hist.axis.style['showValues'] = 0
        self.hist.axis.style['tickAlpha'] = 0
        self.hist.axis.logMode = 1
        # self.hist.plot.opts['antialias'] = 1
        self.hist.setLevels(min=0, max=255)
        
        # policy = QtWidgets.QSizePolicy()
        # policy.setRetainSizeWhenHidden(True)
        # self.hist.setSizePolicy(policy)
        
        # self.histmap_img = self.hist.saveState()
    
    
    def set_tick_hover_color(self):
        for tick in self.hist.gradient.ticks:
            tick.hoverPen = pg.mkPen(self.accent,width=2)
            
    def set_button_color(self):
        color = '#eeeeee' if self.darkmode else '#888888'
        self.ModelButton.setStyleSheet('border: 2px solid {};'.format(color))
            
    def set_crosshair_colors(self):
        pen = pg.mkPen(self.accent)
        self.vLine.setPen(pen)
        self.hLine.setPen(pen)
        [l.setPen(pen) for l in self.vLineOrtho]
        [l.setPen(pen) for l in self.hLineOrtho]
            
    def add_orthoviews(self):
        self.yortho = self.Ly//2
        self.xortho = self.Lx//2
        if self.NZ > 1:
            self.update_ortho()

        self.win.addItem(self.pOrtho[0], 0, 1, rowspan=1, colspan=1)
        self.win.addItem(self.pOrtho[1], 1, 0, rowspan=1, colspan=1)

        qGraphicsGridLayout = self.win.ci.layout
        qGraphicsGridLayout.setColumnStretchFactor(0, 2)
        qGraphicsGridLayout.setColumnStretchFactor(1, 1)
        qGraphicsGridLayout.setRowStretchFactor(0, 2)
        qGraphicsGridLayout.setRowStretchFactor(1, 1)
        
        #self.p0.linkView(self.p0.YAxis, self.pOrtho[0])
        #self.p0.linkView(self.p0.XAxis, self.pOrtho[1])
        
        self.pOrtho[0].setYRange(0,self.Lx)
        self.pOrtho[0].setXRange(-self.dz/3,self.dz*2 + self.dz/3)
        self.pOrtho[1].setYRange(-self.dz/3,self.dz*2 + self.dz/3)
        self.pOrtho[1].setXRange(0,self.Ly)
        # self.pOrtho[0].setLimits(minXRange=self.dz*2+self.dz/3*2)
        # self.pOrtho[1].setLimits(minYRange=self.dz*2+self.dz/3*2)

        self.p0.addItem(self.vLine, ignoreBounds=False)
        self.p0.addItem(self.hLine, ignoreBounds=False)
        self.p0.setYRange(0,self.Lx)
        self.p0.setXRange(0,self.Ly)

        self.win.show()
        self.show()
        
        #self.p0.linkView(self.p0.XAxis, self.pOrtho[1])
        
    def remove_orthoviews(self):
        self.win.removeItem(self.pOrtho[0])
        self.win.removeItem(self.pOrtho[1])
        self.p0.removeItem(self.vLine)
        self.p0.removeItem(self.hLine)
        
        # restore the layout
        qGraphicsGridLayout = self.win.ci.layout
        qGraphicsGridLayout.setColumnStretchFactor(1, 0)
        qGraphicsGridLayout.setColumnStretchFactor(0, 1)
        qGraphicsGridLayout.setRowStretchFactor(1, 0)
        qGraphicsGridLayout.setRowStretchFactor(0, 1)
        
        #restore scale
        self.recenter()
        
        self.win.show()
        self.show()

    def toggle_ortho(self):
        if self.orthobtn.isChecked():
            self.add_orthoviews()
        else:
            self.remove_orthoviews()
            
    def recenter(self):
        buffer = 10 # leave some space between histogram and image
        dy = self.Ly+buffer
        dx = self.Lx
        
        # make room for scale disk
        if self.ScaleOn.isChecked():
            dy += self.pr
            
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

    def reset(self):
        # ---- start sets of points ---- #
        self.selected = 0
        self.X2 = 0
        self.resize = -1
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
        
        # self.initialize_seg()
        # print('reset',self.outpix.shape,self.affinity_graph.shape)
        
        self.ismanual = np.zeros(0, 'bool')
        self.accent = self.palette().brush(QPalette.ColorRole.Highlight).color()
        self.update_plot()
        self.progress.setValue(0)
        self.orthobtn.setChecked(False)
        self.filename = []
        self.loaded = False
        self.recompute_masks = False
        
        
        
 
        

    def initialize_seg(self, compute_affinity=False):
        self.shape = self.masks.shape
        self.dim = len(self.shape) 
        self.steps, self.inds, self.idx, self.fact, self.sign = utils.kernel_setup(self.dim)
        self.supporting_inds = utils.get_supporting_inds(self.steps)
        self.coords = misc.generate_flat_coordinates(self.shape)
        self.neighbors = utils.get_neighbors(self.coords, self.steps, self.dim, self.shape)
        self.indexes, self.neigh_inds, self.ind_matrix = utils.get_neigh_inds(tuple(self.neighbors),self.coords,self.shape)
        self.non_self = np.array(list(set(np.arange(len(self.steps)))-{self.inds[0][0]})) 
        if not hasattr(self,'affinity_graph') or compute_affinity:
            logger.info('initializing affinity graph')
            if np.any(self.masks):
                self.affinity_graph = core.masks_to_affinity(self.masks, self.coords, self.steps, 
                                                       self.inds, self.idx, self.fact, 
                                                       self.sign, self.dim)
            else:
                self.affinity_graph = np.zeros(self.neighbors.shape[1:],bool)

        
    def autosave_on(self):
        if self.SCheckBox.isChecked():
            self.autosave = True
        else:
            self.autosave = False

    def cross_hairs(self):
        if self.CHCheckBox.isChecked():
            self.p0.addItem(self.vLine, ignoreBounds=True)
            self.p0.addItem(self.hLine, ignoreBounds=True)
        else:
            self.p0.removeItem(self.vLine)
            self.p0.removeItem(self.hLine)

    def clear_all(self):
        self.save_state()
        self.prev_selected = 0
        self.selected = 0
        self.layerz = np.zeros((self.Ly,self.Lx,4), np.uint8)
        self.cellpix = np.zeros((self.NZ,self.Ly,self.Lx), np.uint32)
        self.outpix = np.zeros((self.NZ,self.Ly,self.Lx), np.uint32)
        # self.cellcolors = np.array([255,255,255])[np.newaxis,:]
        self.ncells = 0
        # self.toggle_removals()
        self.update_layer()


    def mouse_moved(self, pos):
        items = self.win.scene().items(pos)
        for x in items: #why did this get deleted in CP2?
            if x==self.p0:
                mousePoint = self.p0.mapSceneToView(pos)
                if self.CHCheckBox.isChecked():
                    self.vLine.setPos(mousePoint.x())
                    self.hLine.setPos(mousePoint.y())
        #for x in items:
        #    if not x==self.p0:
        #        QtWidgets.QApplication.restoreOverrideCursor()
        #        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.DefaultCursor)


    def color_choose(self):
        
        # old version forces colormap to onyl apply to image
        # self.color = self.RGBDropDown.currentIndex()
        # self.view = 0
        # self.RGBChoose.button(self.view).setChecked(True)
        
        #new version allows users to select any color map and save it
        # state = self.state[self.view]
        
        self.hist.gradient.loadPreset(self.cmaps[self.RGBDropDown.currentIndex()])
        self.states[self.view] = self.hist.saveState()
        self.set_tick_hover_color()
        # self.update_plot()

    def update_ztext(self):
        zpos = self.currentZ
        try:
            zpos = int(self.zpos.text())
        except:
            logger.warning('ERROR: zposition is not a number')
        self.currentZ = max(0, min(self.NZ-1, zpos))
        self.zpos.setText(str(self.currentZ))
        self.scroll.setValue(self.currentZ)
    
    def update_shape(self): 
        self.Ly, self.Lx, _ = self.stack[self.currentZ].shape
        self.shape = (self.Ly, self.Lx)

    def update_plot(self):
        self.update_shape()
        
        # toggle off histogram for flow field 
        if self.view==1:
            self.opacity_effect.setOpacity(0.0)  # Hide the histogram
            # self.hist.gradient.setEnabled(False)
            # self.hist.region.setEnabled(False)
            # self.hist.background = None
            self.hist.show_histogram = False
            # self.hist.fillLevel = None


        else:
            self.opacity_effect.setOpacity(1.0)  # Show the histogram
            # self.hist.gradient.setEnabled(True)
            # self.hist.region.setEnabled(True)
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
       
        self.scale.setImage(self.radii, autoLevels=False)
        self.scale.setLevels([0.0,255.0])
        #self.img.set_ColorMap(self.bwr)
        if self.NZ>1 and self.orthobtn.isChecked():
            self.update_ortho()
        
        # self.slider.setLow(self.saturation[self.currentZ][0])
        # self.slider.setHigh(self.saturation[self.currentZ][1])
        # if self.masksOn or self.outlinesOn:
        #     self.layer.setImage(self.layerz[self.currentZ], autoLevels=False) <<< something to do with it 
        self.win.show()
        self.show()



    def update_roi_count(self):
        self.roi_count.setText(f'{self.ncells} ROIs')        

    def update_ortho(self):
        if self.NZ>1 and self.orthobtn.isChecked():
            dzcurrent = self.dz
            # self.dz = min(100, max(3,int(self.dzedit.text() )))
            self.dz = min(self.NZ,max(1,int(self.dzedit.text())))
            
            self.zaspect = max(0.01, min(100., float(self.zaspectedit.text())))
            self.dzedit.setText(str(self.dz))
            self.zaspectedit.setText(str(self.zaspect))
            self.update_crosshairs()
            if self.dz != dzcurrent:
                self.pOrtho[0].setXRange(-self.dz/3,self.dz*2 + self.dz/3)
                self.pOrtho[1].setYRange(-self.dz/3,self.dz*2 + self.dz/3)

            y = self.yortho
            x = self.xortho
            z = self.currentZ
            # zmin, zmax = max(0, z-self.dz), min(self.NZ, z+self.dz)
            
            zmin = z-self.dz
            zmax = z+self.dz
            zpad = np.array([0,0])
            if zmin<0:
                zpad[0] = 0-zmin
                zmin = 0 
            if zmax>self.NZ:
                zpad[1] = zmax-self.NZ
                zmax = self.NZ
                
            # to keep ortho view centered on Z, the slice needs to be padded.
            # Zmin and zmax need residuals to padd the array.
            # at present, the cursor on the slice views is orented so that the layer at right/below corresponds to the central view
            b = [0,0]
            if self.view==0:
                for j in range(2):
                    if j==0:
                        image = np.pad(self.stack[zmin:zmax, :, x],(zpad,b,b)).transpose(1,0,2)
                    else:
                        image = np.pad(self.stack[zmin:zmax, y, :],(zpad,b,b))
                        
                    if self.color==0:
                        if self.onechan:
                            # show single channel
                            image = image[...,0]
                        self.imgOrtho[j].setImage(image, autoLevels=False, lut=None)
                    elif self.color>0 and self.color<4:
                        image = image[...,self.color-1]
                        self.imgOrtho[j].setImage(image, autoLevels=False, lut=self.cmap[self.color])
                    elif self.color==4:
                        image = image.astype(np.float32).mean(axis=-1).astype(np.uint8)
                        self.imgOrtho[j].setImage(image, autoLevels=False, lut=None)
                    elif self.color==5:
                        image = image.astype(np.float32).mean(axis=-1).astype(np.uint8)
                        self.imgOrtho[j].setImage(image, autoLevels=False, lut=self.cmap[0])
                    # self.imgOrtho[j].setLevels(self.saturation[self.currentZ])
                self.pOrtho[0].setAspectLocked(lock=True, ratio=self.zaspect)
                self.pOrtho[1].setAspectLocked(lock=True, ratio=1./self.zaspect)

            else:
                image = np.zeros((10,10), np.uint8)
                self.img.setImage(image, autoLevels=False, lut=None)
                self.img.setLevels([0.0, 255.0])        
        self.win.show()
        self.show()

    def update_crosshairs(self):
        self.yortho = min(self.Ly-1, max(0, int(self.yortho)))
        self.xortho = min(self.Lx-1, max(0, int(self.xortho)))
        self.vLine.setPos(self.xortho)
        self.hLine.setPos(self.yortho)
        self.vLineOrtho[1].setPos(self.xortho)
        self.hLineOrtho[1].setPos(self.dz)
        self.vLineOrtho[0].setPos(self.dz)
        self.hLineOrtho[0].setPos(self.yortho)
            

    def compute_scale(self):
        self.diameter = float(self.Diameter.text())
        self.pr = int(float(self.Diameter.text()))
        self.radii_padding = int(self.pr*1.25)
        self.radii = np.zeros((self.Ly+self.radii_padding,self.Lx,4), np.uint8)
        yy,xx = disk([self.Ly+self.radii_padding/2-1, self.pr/2+1],
                            self.pr/2, self.Ly+self.radii_padding, self.Lx)
        # rgb(150,50,150)
        self.radii[yy,xx,0] = 255 # making red to correspond to tooltip
        self.radii[yy,xx,1] = 0
        self.radii[yy,xx,2] = 0
        self.radii[yy,xx,3] = 255
        # self.update_plot()
        self.p0.setYRange(0,self.Ly+self.radii_padding)
        self.p0.setXRange(0,self.Lx)
        self.win.show()
        self.show()

    def set_nchan(self):
        self.nchan = int(self.ChanNumber.text())

    def draw_layer(self, region=None, z=None):
        """
        Re-colorize the overlay (self.layerz) based on self.cellpix[z].
        If region is None, update the entire image. Otherwise, only update
        the specified sub-region: (x_min, x_max, y_min, y_max).
        """
        
        # if region is None:
        #     print('drawing entire layer')
            
        if z is None:
            z = self.currentZ

        # Default to the entire image if region is None
        if region is None:
            region = (0, self.Lx, 0, self.Ly)
            
        x_min, x_max, y_min, y_max = region

        # Clip the region to image bounds
        x_min = max(0, x_min)
        x_max = min(self.Lx, x_max)
        y_min = max(0, y_min)
        y_max = min(self.Ly, y_max)

        # Ensure self.layerz is allocated and correct shape
        if getattr(self, 'layerz', None) is None or self.layerz.shape[:2] != (self.Ly, self.Lx):
            self.layerz = np.zeros((self.Ly, self.Lx, 4), dtype=np.uint8)
        
        # Extract subarray of cellpix
        sub_cellpix = self.cellpix[z, y_min:y_max, x_min:x_max]

        # Prepare a subarray for color
        sub_h = y_max - y_min
        sub_w = x_max - x_min
        sub_layerz = np.zeros((sub_h, sub_w, 4), dtype=np.uint8)
        # 1) Color + Alpha
        if self.masksOn and self.view == 0:
            # Basic coloring
            sub_layerz[..., :3] = self.cellcolors[sub_cellpix, :] if len(self.cellcolors) > 1 else [255,0,0]
            sub_layerz[..., 3] = self.opacity * (sub_cellpix > 0).astype(np.uint8)

            # Selected cell -> white
            if self.selected > 0:
                mask_sel = (sub_cellpix == self.selected)
                sub_layerz[mask_sel] = np.array([255, 255, 255, self.opacity], dtype=np.uint8)
        else:
            # No masks -> alpha=0
            sub_layerz[..., 3] = 0

        # 2) Outlines
        if self.outlinesOn:
            # there is something weird going on woith initializing the affinity graoh from the npy
            # they need to be deleted I think, or need some workaround to overwrite masks and shape etc. 
            # as they get reset as 512
            
            
            # print(self.cellpix[z].shape, self.shape,self.affinity_graph.shape, len(self.coords), self.coords[0].shape)
            # self.outpix = core.affinity_to_boundary( self.cellpix[z], self.affinity_graph, tuple(self.coords))[np.newaxis,:,:]
            sub_outpix = self.outpix[z, y_min:y_max, x_min:x_max]
            sub_layerz[sub_outpix > 0] = np.array(self.outcolor, dtype=np.uint8)

        # Put the subarray back into the main overlay
        self.layerz[y_min:y_max, x_min:x_max] = sub_layerz

        # Finally update the displayed image
        self.layer.setImage(self.layerz, autoLevels=False)

    def compute_saturation(self):
        # compute percentiles from stack
        self.saturation = []
        for n in range(len(self.stack)):
            # reverted for cellular images, maybe there can be an option?
            vals = self.slider.value()

            self.saturation.append([np.percentile(self.stack[n].astype(np.float32),vals[0]),
                                    np.percentile(self.stack[n].astype(np.float32),vals[1])])
            
    def chanchoose(self, image):
        if image.ndim > 2 and not self.onechan:
            if self.ChannelChoose[0].currentIndex()==0:
                image = image.astype(np.float32).mean(axis=-1)[...,np.newaxis]
            else:
                chanid = [self.ChannelChoose[0].currentIndex()-1]
                if self.ChannelChoose[1].currentIndex()>0:
                    chanid.append(self.ChannelChoose[1].currentIndex()-1)
                image = image[:,:,chanid].astype(np.float32)
        return image

    # Looks like CP2 might not do net averaging in the GUI, also defaults to torch
    # The CP2 version breaks omnipose, something to do with those extra if/else that
    # correspond to extra cases that the models() function already takes case of 
    
#     def get_model_path(self):
#         self.current_model = self.ModelChoose.currentText()
#         self.current_model_path = os.fspath(models.MODEL_DIR.joinpath(self.current_model))
        
#     def initialize_model(self, model_name=None):
#         if model_name is None or not isinstance(model_name, str):
#             self.get_model_path()
#             self.model = models.CellposeModel(gpu=self.useGPU.isChecked(), 
#                                               pretrained_model=self.current_model_path)
#         else:
#             self.current_model = model_name
#             if 'cyto' in self.current_model or 'nuclei' in self.current_model:
#                 self.current_model_path = models.model_path(self.current_model, 0)
#             else:
#                 self.current_model_path = os.fspath(models.MODEL_DIR.joinpath(self.current_model))
#             if self.current_model=='cyto':
#                 self.model = models.Cellpose(gpu=self.useGPU.isChecked(), 
#                                              model_type=self.current_model)
#             else:
#                 self.model = models.CellposeModel(gpu=self.useGPU.isChecked(), 
#                                                   model_type=self.current_model)

    def get_model_path(self):
        self.current_model = self.ModelChoose.currentText()
        if self.current_model in models.MODEL_NAMES:
            self.current_model_path = models.model_path(self.current_model, 0, self.torch)
        else:
            self.current_model_path = os.fspath(models.MODEL_DIR.joinpath(self.current_model))
        
    # this is where we need to change init depending on whether or not we have a size model
    # or which size model to use... 
    # this should really be updated to allow for custom size models to be used, too
    # I guess most doing that will not be using the GUI, but still an important feature 
    def initialize_model(self):
        self.get_model_path()


        if self.current_model in models.MODEL_NAMES:

            # make sure 2-channel models are initialized correctly
            if self.current_model in models.C2_MODEL_NAMES:
                self.nchan = 2
                self.ChanNumber.setText(str(self.nchan))

            # ensure that the boundary/nclasses is set correctly
            self.boundary.setChecked(self.current_model in models.BD_MODEL_NAMES)
            self.nclasses = 2 + self.boundary.isChecked()

            logger.info(f'Initializing model: nchan set to {self.nchan}, nclasses set to {self.nclasses}, dim set to {self.dim}')        

            if self.SizeModel.isChecked():
                self.model = models.Cellpose(gpu=self.useGPU.isChecked(),
                                             use_torch=self.torch,
                                             model_type=self.current_model,
                                             nchan=self.nchan,
                                             nclasses=self.nclasses)
            else:
                self.model = models.CellposeModel(gpu=self.useGPU.isChecked(),
                                                  use_torch=self.torch,
                                                  model_type=self.current_model,                                             
                                                  nchan=self.nchan,
                                                  nclasses=self.nclasses)
            
            omni_model = 'omni' in self.current_model
            bacterial = 'bact' in self.current_model
            if omni_model or bacterial:
                self.NetAvg.setCurrentIndex(1) #one run net
                
        else:
            self.nclasses = 2 + self.boundary.isChecked()
            self.model = models.CellposeModel(gpu=self.useGPU.isChecked(), 
                                              use_torch=True,
                                              pretrained_model=self.current_model_path,                                             
                                              nchan=self.nchan,
                                              nclasses=self.nclasses)

            
    def add_model(self):
        io._add_model(self)
        return

    def remove_model(self):
        io._remove_model(self)
        return

    def new_model(self):
        if self.NZ!=1:
            logger.error('cannot train model on 3D data')
            return
        
        # train model
        image_names = self.get_files()[0]
        self.train_data, self.train_labels, self.train_files = io._get_train_set(image_names)
        TW = guiparts.TrainWindow(self, models.MODEL_NAMES)
        train = TW.exec_()
        if train:
            logger.info(f'training with {[os.path.split(f)[1] for f in self.train_files]}')
            self.train_model()

        else:
            logger.info('training cancelled')

    # this probably needs an overhaul 
    def train_model(self):
        if self.training_params['model_index'] < len(models.MODEL_NAMES):
            model_type = models.MODEL_NAMES[self.training_params['model_index']]
            logger.info(f'training new model starting at model {model_type}')        
        else:
            model_type = None
            logger.info(f'training new model starting from scratch')     
        self.current_model = model_type   
        
        self.channels = self.get_channels()
        logger.info(f'training with chan = {self.ChannelChoose[0].currentText()}, chan2 = {self.ChannelChoose[1].currentText()}')
            
        self.model = models.CellposeModel(gpu=self.useGPU.isChecked(), 
                                          model_type=model_type)

        save_path = os.path.dirname(self.filename)
        
        logger.info('name of new model:' + self.training_params['model_name'])
        self.new_model_path = self.model.train(self.train_data, self.train_labels, 
                                               channels=self.channels, 
                                               save_path=save_path, 
                                               nimg_per_epoch=8,
                                               learning_rate = self.training_params['learning_rate'], 
                                               weight_decay = self.training_params['weight_decay'], 
                                               n_epochs = self.training_params['n_epochs'],
                                               model_name = self.training_params['model_name'])
        diam_labels = self.model.diam_labels.copy()
        # run model on next image 
        io._add_model(self, self.new_model_path, load_model=False)
        self.new_model_ind = len(self.model_strings)
        self.autorun = True
        if self.autorun:
            channels = self.channels.copy()
            self.clear_all()
            self.get_next_image(load_seg=True)
            # keep same channels
            self.ChannelChoose[0].setCurrentIndex(channels[0])
            self.ChannelChoose[1].setCurrentIndex(channels[1])
            self.diameter = diam_labels
            self.Diameter.setText('%0.2f'%self.diameter)        
            logger.info(f'>>>> diameter set to diam_labels ( = {diam_labels: 0.3f} )')
            self.compute_model()
        logger.info(f'!!! computed masks for {os.path.split(self.filename)[1]} from new model !!!')
        
    def get_thresholds(self):
        # the text field version
        # also, the special case for NZ>1 desn't make sense for omnipose
        # which can calculate flows in 3D 
        
#         try:
#             flow_threshold = float(self.flow_threshold.text())
#             cellprob_threshold = float(self.cellprob_threshold.text())
#             if flow_threshold==0.0 or self.NZ>1:
#                 flow_threshold = None
                
#             return flow_threshold, cellprob_threshold

        #The slider version 
        try:
            return self.threshslider.value(), self.probslider.value()
        except Exception as e:
            print('flow threshold or cellprob threshold not a valid number, setting to defaults')
            self.flow_threshold.setText('0.0')
            self.cellprob_threshold.setText('0.0')
            return 0.0, 0.0

    def run_mask_reconstruction(self):
        # use_omni = 'omni' in self.current_model
        
        # needed to be replaced with recompute_masks
        # rerun = False
        have_enough_px = self.probslider.value() > self.cellprob # slider moves up
        print('debug',have_enough_px,self.probslider.value(),self.cellprob)
        
        # update thresholds
        self.threshold, self.cellprob = self.get_thresholds()

        
        # if self.cellprob != self.probslider.value():
        #     rerun = True
        #     self.cellprob = self.probslider.value()
            
        # if self.threshold != self.threshslider.value():
        #     rerun = True
        #     self.threshold = self.threshslider.value()
        
        # if not self.recompute_masks:
        #     return
        
        self.threshold, self.cellprob = self.get_thresholds()
        
        if self.threshold is None:
            logger.info('computing masks with cell prob=%0.3f, no flow error threshold'%
                    (self.cellprob))
        else:
            logger.info('computing masks with cell prob=%0.3f, flow error threshold=%0.3f'%
                    (self.cellprob, self.threshold))

        net_avg = self.NetAvg.currentIndex()==0 and self.current_model in models.MODEL_NAMES
        resample = self.NetAvg.currentIndex()<2
        omni = OMNI_INSTALLED and self.omni.isChecked()
        
        # useful printout for easily copying parameters to a notebook etc. 
        s = ('channels={}, mask_threshold={:.2f}, '
             'flow_threshold={:.2f}, diameter={:.2f}, invert={}, cluster={}, net_avg={},'
             'do_3D={}, omni={}'
            ).format(self.get_channels(),
                     self.cellprob,
                     self.threshold,
                     self.diameter,
                     self.invert.isChecked(),
                     self.cluster.isChecked(),
                     net_avg,
                     False,
                     omni)
        
        self.runstring.setPlainText(s)
            
        if not omni:
            maski = dynamics.compute_masks(dP=self.flows[-1][:-1], 
                                           cellprob=self.flows[-1][-1],
                                           p=self.flows[-2].copy(),  
                                           mask_threshold=self.cellprob,
                                           flow_threshold=self.threshold,
                                           resize=self.cellpix.shape[-2:],
                                           verbose=self.verbose.isChecked())[0]
        else:
            #self.flows[3] is p, self.flows[-1] is dP, self.flows[5] is dist/prob, self.flows[6] is bd
            
            # must recompute flows trajectory if we add pixels, because p does not contain them
            # an alternate approach would be to compute p for the lowest allowed threshold
            # and then never recompute (the threshold prodces a mask that selects from existing trajectories, see get_masks)
            # seems like the dbscanm method breaks with this, but affinity is fine... 
            # p = self.flows[-2].copy() if have_enough_px  else None 
            p = self.flows[-2].copy() if have_enough_px and self.AffinityCheck.isChecked() else None 
        
            
            dP = self.flows[-1][:-self.model.dim]
            dist = self.flows[-1][self.model.dim]
            bd = self.flows[-1][self.model.dim+1]
            # print('flow debug',self.model.dim,p.shape,dP.shape,dist.shape,bd.shape)
            ret = core.compute_masks(dP=dP, 
                                    dist=dist, 
                                    affinity_graph=None, 
                                    bd=bd,
                                    p=p, 
                                    mask_threshold=self.cellprob,
                                    flow_threshold=self.threshold,
                                    resize=self.cellpix.shape[-2:],
                                    cluster=self.cluster.isChecked(),
                                    verbose=self.verbose.isChecked(),
                                    nclasses=self.model.nclasses,
                                    affinity_seg=self.AffinityCheck.isChecked(),
                                    omni=omni)
            
            maski, p, tr, bounds, augmented_affinity = ret
            
            self.masks = maski
            self.shape = maski.shape
            self.dim = maski.ndim
            # self.neighbors = augmented_affinity[:self.dim]
            # self.affinity_graph = augmented_affinity[self.dim] need to cnvert indexing to update the full affinity
            # or just flatten the spatial affinity 
            coords = np.nonzero(self.masks) 
            self.bounds = core.affinity_to_boundary(self.masks, self.affinity_graph, coords)
            
            # for the pruposes of the GUI, we may want to store the affinity graph as a
            # (8,Y,X) array rather than a (9,N) array... unless N is always the same size 
            
            # slicing is probably going to be a lot easier if we use the spatial affinity format 
            self.spatial_affinity = core.spatial_affinity(self.affinity_graph, self.coords, self.shape)
            
            # self.neighbors = core.get_neighbors(self.spatial_affinity, self.meshgrid, self.shape)
            # self.meshgrid = misc.meshgrid(self.shape)
            # self.indexes, self.neigh_inds, self.ind_matrix = utils.get_neigh_inds(tuple(self.neighbors),self.meshgrid,self.shape)
            
            # self.steps, self.inds, self.idx, self.fact, self.sign = utils.kernel_setup(self.dim)
            # self.non_self = np.array(list(set(np.arange(len(self.steps)))-{self.inds[0][0]})) 

            print('\n\nassgin the affinity tot he main one here \n')
        
        self.masksOn = True
        self.MCheckBox.setChecked(True)
        # self.outlinesOn = True #should not turn outlines back on by default; masks make sense though 
        # self.OCheckBox.setChecked(True)
        if maski.ndim<3:
            maski = maski[np.newaxis,...]
        logger.info('%d cells found'%(len(misc.unique_nonzero(maski))))
        io._masks_to_gui(self) # replace this to show boundary emphasized masks
        self.show()


    def suggest_model(self, model_name=None):
        logger.info('computing styles with 2D image...')
        data = self.stack[self.NZ//2].copy()
        styles_gt = np.load(os.fspath(pathlib.Path.home().joinpath('.cellpose', 'style_choice.npy')), 
                            allow_pickle=True).item()
        train_styles, train_labels, label_models = styles_gt['train_styles'], styles_gt['leiden_labels'], styles_gt['label_models']
        self.diameter = float(self.Diameter.text())
        self.current_model = 'general'
        channels = self.get_channels()
        model = models.CellposeModel(model_type='general', gpu=self.useGPU.isChecked())
        styles = model.eval(data, 
                            channels=channels, 
                            diameter=self.diameter, 
                            compute_masks=False)[-1]

        n_neighbors = 5
        dists = ((train_styles - styles)**2).sum(axis=1)**0.5
        neighbor_labels = train_labels[dists.argsort()[:n_neighbors]]
        label = mode(neighbor_labels)[0][0]
        model_type = label_models[label]
        logger.info(f'style suggests model {model_type}')
        ind = self.net_text.index(model_type)
        # for i in range(len(self.net_text)):
        #     self.StyleButtons[i].setStyleSheet(self.styleUnpressed)
        # self.StyleButtons[ind].setStyleSheet(self.stylePressed)
        self.compute_model(model_name=model_type)
            
    def compute_model(self):
        self.progress.setValue(10)
        QApplication.processEvents() 
        try:
            tic=time.time()
            self.clear_all()
            self.flows = [[],[],[]]
            self.initialize_model()
            logger.info('using model %s'%self.current_model)
            self.progress.setValue(20)
            QApplication.processEvents() 
            do_3D = False
            if self.NZ > 1:
                do_3D = True
                data = self.stack.copy()
            else:
                data = self.stack[0].copy() # maybe chanchoose here 
            channels = self.get_channels()
            
            self.diameter = float(self.Diameter.text())
            
            
            ### will either have to put in edge cases for worm etc or just generalize model loading to respect what is there 
            try:
                omni_model = 'omni' in self.current_model
                bacterial = 'bact' in self.current_model
                if omni_model or bacterial:
                    self.NetAvg.setCurrentIndex(1) #one run net
                # if bacterial:
                #     self.diameter = 0.
                #     self.Diameter.setText('%0.1f'%self.diameter)

                # allow omni to be togged manually or forced by model
                if OMNI_INSTALLED:
                    if omni_model:
                        logger.info('turning on Omnipose mask recontruction version for Omnipose models (see menu)')
                        if not self.omni.isChecked():
                            print('WARNING: Omnipose models require Omnipose mask recontruction (toggle back on in menu)')
                        if not self.cluster.isChecked():
                            print(('NOTE: clutering algorithm can help with over-segmentation in thin cells.'
                                   'Default is ON with omnipose models (see menu)'))
                            
                    elif self.omni.isChecked():
                        print('NOTE: using Omnipose mask recontruction with built-in cellpose model (toggle in Omnipose menu)')

                net_avg = self.NetAvg.currentIndex()==0 and self.current_model in models.MODEL_NAMES
                resample = self.NetAvg.currentIndex()<2
                omni = OMNI_INSTALLED and self.omni.isChecked()
                
                self.threshold, self.cellprob = self.get_thresholds()

                # useful printout for easily copying parameters to a notebook etc. 
                s = ('channels={}, mask_threshold={}, '
                     'flow_threshold={}, diameter={}, invert={}, cluster={}, net_avg={}, '
                     'do_3D={}, omni={}'
                    ).format(self.get_channels(),
                             self.cellprob,
                             self.threshold,
                             self.diameter,
                             self.invert.isChecked(),
                             self.cluster.isChecked(),
                             net_avg,
                             do_3D,
                             omni)
                self.runstring.setPlainText(s)
                self.progress.setValue(30)
                
                masks, flows = self.model.eval(data, channels=channels,
                                               mask_threshold=self.cellprob,
                                               flow_threshold=self.threshold,
                                               diameter=self.diameter, 
                                               invert=self.invert.isChecked(),
                                               net_avg=net_avg, 
                                               augment=False, 
                                               resample=resample,
                                               do_3D=do_3D, 
                                               progress=self.progress,
                                               verbose=self.verbose.isChecked(),
                                               omni=omni, 
                                               tile=False,
                                               affinity_seg=self.AffinityCheck.isChecked(),
                                               cluster = self.cluster.isChecked(),
                                               transparency=True,
                                               channel_axis=-1
                                               )[:2]
                
            except Exception as e:
                print('GUI.py: NET ERROR: %s'%e)
                self.progress.setValue(0)
                return
            
            self.progress.setValue(75)
            QApplication.processEvents() 
            #if not do_3D:
            #    masks = masks[0][np.newaxis,:,:]
            #    flows = flows[0]
            
            # flows here are [RGB, dP, cellprob, p, bd, tr]
            self.flows[0] = to_8_bit(flows[0]) #RGB flow for plotting
            self.flows[1] = to_8_bit(flows[2]) #dist/prob for plotting
            if self.boundary.isChecked():
                self.flows[2] = to_8_bit(flows[4]) #boundary for plotting
            else:
                self.flows[2] = np.zeros_like(self.flows[1])
                
            # boundary and affinity
            self.bounds = flows[-1]
            self.masks = masks
            
            
            affinity = flows[-2]
            print(len(affinity))
            if not len(affinity):
                # if the affinity graph is returned empty, we can just recompute it
                self.initialize_seg(compute_affinity=True)
            else:
                # self.update_affinity(affiniyt)
                inds = self.ind_matrix[self.masks>0]
                self.affinity_graph = np.zeros(self.neighbors.shape[1:],bool)
                self.affinity_graph[:,inds] = affinity
                
            # print('affiniyt graph',flows[-2].shape)
            # self.affinity_graph = flows[-2]
            
            
            
            print('run assign here too')

            if not do_3D:
                masks = masks[np.newaxis,...]
                for i in range(3):
                    self.flows[i] = resize_image(self.flows[i], masks.shape[-2], masks.shape[-1])
               
                #critical line from what I had commended out below
                self.flows = [self.flows[n][np.newaxis,...] for n in range(len(self.flows))]
            
            # I think this is a z-component placeholder. Relaceing with boundary output, will
            # put this back later for the 3D update 
            # if not do_3D:
            #     self.flows[2] = np.zeros(masks.shape[1:], dtype=np.uint8)
            #     self.flows = [self.flows[n][np.newaxis,...] for n in range(len(self.flows))]
            # else:
            #     self.flows[2] = (flows[1][0]/10 * 127 + 127).astype(np.uint8)
                

            # this stores the original flow components for recomputing masks
            if len(flows)>2: 
                self.flows.append(flows[3].squeeze()) #p put in position -2
                flws = [flows[1], #self.flows[-1][:self.dim] is dP
                        flows[2][np.newaxis,...]] #self.flows[-1][self.dim] is dist/prob
                if self.boundary.isChecked():
                    flws.append(flows[4][np.newaxis,...]) #self.flows[-1][self.dim+1] is bd
                else:
                    flws.append(np.zeros_like(flws[-1]))
                
                self.flows.append(np.concatenate(flws))

            logger.info('%d cells found with model in %0.3f sec'%(len(np.unique(masks)[1:]), time.time()-tic))
            self.progress.setValue(80)
            QApplication.processEvents() 
            z=0
            self.masksOn = True
            self.MCheckBox.setChecked(True)
            # self.outlinesOn = True #again, this option should persist and not get toggled by another GUI action 
            # self.OCheckBox.setChecked(True)

            # print('masks found, drawing now', self.masks.shape)
            io._masks_to_gui(self)
            self.progress.setValue(100)

            # self.toggle_server(off=True)
            if not do_3D:
                self.threshslider.setEnabled(True)
                self.probslider.setEnabled(True)
        except Exception as e:
            print('ERROR: %s'%e)

    def copy_runstring(self):
        self.clipboard.setText(self.runstring.toPlainText())

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
        # self.SCheckBox.setEnabled(True)
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

    def toggle_mask_ops(self):
        self.toggle_removals()

        
        
