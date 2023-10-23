import signal, sys, os, pathlib, warnings, datetime, tempfile, glob, time
import gc
from natsort import natsorted
from tqdm import tqdm, trange

from PyQt6 import QtGui, QtCore, QtWidgets, QtSvgWidgets
from PyQt6.QtCore import Qt, QTimer, pyqtSlot, QCoreApplication
from PyQt6.QtWidgets import QMainWindow, QApplication, QSizePolicy, QWidget, QScrollBar, QSlider, QComboBox, QGridLayout, QPushButton, QFrame, QCheckBox, QLabel, QProgressBar, QLineEdit, QMessageBox, QGroupBox, QDoubleSpinBox, QPlainTextEdit, QScrollArea
from PyQt6.QtGui import QColor, QPalette
import pyqtgraph as pg
# from pyqtgraph import GraphicsScene

os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '1'

import numpy as np
from scipy.stats import mode
import cv2
from scipy.ndimage import gaussian_filter

from . import guiparts, menus, io
from .. import models, core, dynamics
from ..utils import download_url_to_file, masks_to_outlines, diameters 
from ..io import save_server, get_image_files, imsave, imread, check_dir #OMNI_INSTALLED
from ..transforms import resize_image #fixed import
from ..plot import disk
from omnipose.utils import normalize99, to_8_bit

from .guiparts import TOOLBAR_WIDTH, SPACING, WIDTH_0

ALLOWED_THEMES = ['light','dark']

INPUT_WIDTH = 2*WIDTH_0 + SPACING
WIDTH_3 = 3*WIDTH_0+2*SPACING
WIDTH_5 = 5*WIDTH_0+4*SPACING

import darkdetect
import qdarktheme
import superqt

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB = True
except:
    MATPLOTLIB = False


try:
    from google.cloud import storage
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                        'key/cellpose-data-writer.json')
    SERVER_UPLOAD = True
except:
    SERVER_UPLOAD = False

OMNI_INSTALLED = 1
if OMNI_INSTALLED:
    import omnipose
    from omnipose.utils import normalize99 # replace the cellpose version to avoid low-cell-density artifacts
    
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
    # DEFAULT_MODEL = 'cyto2'
    

    from omnipose.utils import sinebow
    # from colour import rgb2hex
    from matplotlib.colors import rgb2hex

    N = 29
    c = sinebow(N)
    COLORS = [rgb2hex(c[i][:3]) for i in range(1,N+1)] #can only do RBG, not RGBA for stylesheet

else:
    PRELOAD_IMAGE = None # could make this once from cyto 
    DEFAULT_MODEL = 'cyto2'
    cp_dir = pathlib.Path.home().joinpath('.cellpose')
    check_dir(cp_dir)
    ICON_PATH = pathlib.Path.home().joinpath('.cellpose', 'logo.png')
    ICON_URL = 'https://www.cellpose.org/static/images/cellpose_transparent.png'
    COLORS = ['ff0000']*4

if not ICON_PATH.is_file():
    print('downloading logo from', ICON_URL,'to', ICON_PATH)
    download_url_to_file(ICON_URL, ICON_PATH, progress=True)

# Not everyone with have a math font installed, so all this effort just to have
# a cute little math-style gamma as a slider label...
GAMMA_PATH = pathlib.Path.home().joinpath('.omnipose','gamma.svg')
GAMMA_URL = 'https://github.com/kevinjohncutler/omnipose/blob/main/gui/gamma.svg?raw=true'   
if not GAMMA_PATH.is_file():
    print('downloading gamma icon from', GAMMA_URL,'to', GAMMA_PATH)
    download_url_to_file(GAMMA_URL, GAMMA_PATH, progress=True)
    

#Define possible models; can we make a master list in another file to use in models and main? 

def checkstyle(color):
    return ''.join(['QCheckBox::indicator{',
            'color : {};'.format(color),
            '}'])

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

def avg3d(C):
    """ smooth value of c across nearby points
        (c is center of grid directly below point)
        b -- a -- b
        a -- c -- a
        b -- a -- b
    """
    Ly, Lx = C.shape
    # pad T by 2
    T = np.zeros((Ly+2, Lx+2), np.float32)
    M = np.zeros((Ly, Lx), np.float32)
    T[1:-1, 1:-1] = C.copy()
    y,x = np.meshgrid(np.arange(0,Ly,1,int), np.arange(0,Lx,1,int), indexing='ij')
    y += 1
    x += 1
    a = 1./2 #/(z**2 + 1)**0.5
    b = 1./(1+2**0.5) #(z**2 + 2)**0.5
    c = 1.
    M = (b*T[y-1, x-1] + a*T[y-1, x] + b*T[y-1, x+1] +
         a*T[y, x-1]   + c*T[y, x]   + a*T[y, x+1] +
         b*T[y+1, x-1] + a*T[y+1, x] + b*T[y+1, x+1])
    M /= 4*a + 4*b + c
    return M

def interpZ(mask, zdraw):
    """ find nearby planes and average their values using grid of points
        zfill is in ascending order
    """
    ifill = np.ones(mask.shape[0], "bool")
    zall = np.arange(0, mask.shape[0], 1, int)
    ifill[zdraw] = False
    zfill = zall[ifill]
    zlower = zdraw[np.searchsorted(zdraw, zfill, side='left')-1]
    zupper = zdraw[np.searchsorted(zdraw, zfill, side='right')]
    for k,z in enumerate(zfill):
        Z = zupper[k] - zlower[k]
        zl = (z-zlower[k])/Z
        plower = avg3d(mask[zlower[k]]) * (1-zl)
        pupper = avg3d(mask[zupper[k]]) * zl
        mask[z] = (plower + pupper) > 0.33
        #Ml, norml = avg3d(mask[zlower[k]], zl)
        #Mu, normu = avg3d(mask[zupper[k]], 1-zl)
        #mask[z] = (Ml + Mu) / (norml + normu)  > 0.5
    return mask, zfill
            
global logger
def run(image=PRELOAD_IMAGE):
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    from ..io import logger_setup
    global logger
    logger, log_file = logger_setup()
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
        print('downloading logo')
        download_url_to_file('https://www.cellpose.org/static/images/cellpose_transparent.png', icon_path, progress=True)
    if not guip_path.is_file():
        print('downloading help window image')
        download_url_to_file('https://www.cellpose.org/static/images/cellpose_gui.png', guip_path, progress=True)
    if not style_path.is_file():
        print('downloading style classifier')
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
        win.set_hist_colors()
        win.set_button_color()
        win.set_crosshair_colors()
        # win.update_plot()
    app.paletteChanged.connect(sync_theme_with_system)             
    sync_theme_with_system()

    ret = app.exec()
    sys.exit(ret)


def get_unique_points(set):
    cps = np.zeros((len(set),3), np.int32)
    for k,pp in enumerate(set):
        cps[k,:] = np.array(pp)
    set = list(np.unique(cps, axis=0))
    return set



class MainW(QMainWindow):
    def __init__(self, size, dpi, pxr, clipboard, image=None):
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
        menus.helpmenu(self)
        menus.omnimenu(self)

        self.model_strings = models.MODEL_NAMES.copy()
        
        # self.setStyleSheet("QMainWindow {background:'black';}")
        # self.stylePressed = ''.join(["QPushButton {Text-align: middle; ",
        #                      "background-color: {};".format('#484848'),
        #                      "border-color: #565656;",
        #                      "color:white;}"])
        c = self.palette().brush(QPalette.ColorRole.Base).color()
        text_color = 'rgba'+str(self.palette().brush(QPalette.ColorRole.Text).color().getRgb())
        self.styleUnpressed = ''.join(["QPushButton {Text-align: middle; ",
                                       "background-color: {}; ".format('rgba'+str(c.getRgb())),
                                       "color: {}; ".format(text_color),
                                       "}"])
        c = self.palette().brush(QPalette.ColorRole.Button).color()
        self.stylePressed = ''.join(["QPushButton {Text-align: middle; ",
                                       "background-color: {}; ".format('rgba'+str(c.getRgb())),
                                        "border-color: {}; ".format('rgba'+str(self.palette().brush(QPalette.ColorRole.ButtonText).color().getRgb())),
                                       "color: {}; ".format(text_color),
                                       "}"])
        # self.styleInactive = ("QPushButton {Text-align: middle; "
        #                       "background-color: #303030; "
        #                      "border-color: #565656;"
        #                       "color: #fff;}")
        self.stylePressed = ''
        self.styleUnpressed = '' 
        # c.setAlpha(20)
        # region.setBrush(c) # I hate the blue background getRgb(c)
        self.styleInactive = ''
        self.textbox_style = ''#background-color: black;

        self.loaded = False

        # ---- MAIN WIDGET LAYOUT ---- #

        scrollable = 1 
        if scrollable:
            self.l0 = QGridLayout(self)
            self.scrollArea = QScrollArea(self)
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
        self.l0.addWidget(self.win, 0, TOOLBAR_WIDTH+1, b, 3*b)
        self.win.scene().sigMouseClicked.connect(self.plot_clicked)
        self.win.scene().sigMouseMoved.connect(self.mouse_moved)
        self.make_viewbox()
        self.make_orthoviews()
        self.l0.setColumnStretch(TOOLBAR_WIDTH+1, 1)
        # self.l0.setMaximumWidth(100)
        self.ScaleOn.setChecked(False)  # can only toggle off after make_viewbox is called 

        # hard-coded colormaps entirely replaced with pyqtgraph

        if MATPLOTLIB:
            self.colormap = (plt.get_cmap('gist_ncar')(np.linspace(0.0,.9,1000000)) * 255).astype(np.uint8)
            np.random.seed(42) # make colors stable
            self.colormap = self.colormap[np.random.permutation(1000000)]
        else:
            np.random.seed(42) # make colors stable
            self.colormap = ((np.random.rand(1000000,3)*0.8+0.1)*255).astype(np.uint8)
        

        self.is_stack = True # always loading images of same FOV
        # if called with image, load it
        if image is not None:
            self.filename = image
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
        
        # policy = QtWidgets.QSizePolicy()
        # policy.setRetainSizeWhenHidden(True)
        # self.p0.setSizePolicy(policy)
    
                    
    def help_window(self):
        HW = guiparts.HelpWindow(self)
        HW.show()

    def train_help_window(self):
        THW = guiparts.TrainHelpWindow(self)
        THW.show()

    def gui_window(self):
        EG = guiparts.ExampleGUI(self)
        EG.show()

    def make_buttons(self):
        label_style = ''
        COLORS[0] = '#545454'
        print( self.height(), self.width())
        ftwt1 = self.height()//50
        ftwt2 = self.height()//100
        ftwt3 = self.height()//200
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

        self.headings = ''
        # self.dropdowns = ("QComboBox QAbstractItemView { color: white;"
        #                   "background-color: #303030;"
        #                   "selection-color: white; "
        #                   "min-width: 100px; }")
        #                 # "selection-background-color: rgb(50,100,50);")
        # self.checkstyle = ("color: white;"
        #                   "selection-background-color: {};").format(COLORS[0])
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
        
    
        # b+=4
        
        
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

        
        self.slider.setMinimum(0.0)
        self.slider.setMaximum(100.0)
        self.slider.setValue((0.1,99.9))  
        self.slider._max_label.setFont(self.medfont)
        self.slider._min_label.setFont(self.medfont)
        self.l0.addWidget(self.slider, b,c+1,1,TOOLBAR_WIDTH-(c+1))

        label = QLabel('\u2702') #scissors icon
        label.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
        label.setStyleSheet('color: #888888')
        # self.iconfont = QtGui.QFont("Arial")
        # self.iconfont.setPixelSize(18)
        # self.iconfont.setWeight(QtGui.QFont.Weight.Bold)
        # label.setFont(self.iconfont)
        label.setFont(self.boldfont)

        self.l0.addWidget(label, b,c,1,1)
        
        
        b+=1
        button = QPushButton('')
        button.setIcon(QtGui.QIcon(str(GAMMA_PATH)))
        button.setStyleSheet("QPushButton {Text-align: middle; background-color: none;}")
        button.setDefault(True)
        self.l0.addWidget(button, b,c,1,1)
        
        # self.l0.addWidget(label, b,c,1,1)
        # self.l0.addWidget(svg, b,c,1,1)
        
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
        
        b+=2
        label = QLabel('Drawing:')
        label.setStyleSheet('color: {}'.format(COLORS[0]))
        label.setFont(self.boldfont)
        self.l0.addWidget(label, b,0,1,TOOLBAR_WIDTH)

        b+=1
        c = TOOLBAR_WIDTH//2
        label = QLabel('pen:')
        label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        label.setStyleSheet(label_style)
        label.setFont(self.medfont)
        self.l0.addWidget(label, b, c-1, 1,2)
        c+=1
        self.brush_size = 1
        self.BrushChoose = QComboBox()
        self.BrushChoose.addItems(["off","1px","3px","5px","7px","9px"])
        self.BrushChoose.currentIndexChanged.connect(self.brush_choose)
        # self.BrushChoose.setFixedWidth(60)
        self.BrushChoose.setStyleSheet(self.dropdowns())
        self.BrushChoose.setFont(self.medfont)
        self.BrushChoose.setFixedWidth(WIDTH_3)
        self.BrushChoose.setCurrentIndex(0)
        self.l0.addWidget(self.BrushChoose, b, c,1, TOOLBAR_WIDTH-c)


        # turn off masks
        self.layer_off = False
        self.masksOn = True
        self.MCheckBox = QCheckBox('masks')
        self.MCheckBox.setToolTip('Press X or M to toggle masks')
        self.MCheckBox.setStyleSheet(self.checkstyle)
        self.MCheckBox.setFont(self.medfont)
        self.MCheckBox.setChecked(self.masksOn)
        self.MCheckBox.toggled.connect(self.toggle_masks)
        self.l0.addWidget(self.MCheckBox, b,0,1,2)
        
        # turn on ncolor
        b+=1
        self.ncolor = True
        self.NCCheckBox = QCheckBox('n-color')
        self.NCCheckBox.setToolTip('Press C or N to toggle n-color masks')
        self.NCCheckBox.setStyleSheet(self.checkstyle)
        self.NCCheckBox.setFont(self.medfont)
        self.NCCheckBox.setChecked(self.ncolor)
        self.NCCheckBox.toggled.connect(self.toggle_ncolor)
        self.l0.addWidget(self.NCCheckBox, b, 0,1,2)


        b+=1
        # turn off outlines
        self.outlinesOn = False # turn off by default
        self.OCheckBox = QCheckBox('outlines')
        self.OCheckBox.setToolTip('Press Z or O to toggle outlines')
        self.OCheckBox.setStyleSheet(self.checkstyle)
        self.OCheckBox.setFont(self.medfont)
        self.l0.addWidget(self.OCheckBox, b,0,1,2)
        
        self.OCheckBox.setChecked(False)
        self.OCheckBox.toggled.connect(self.toggle_masks) 
        
        # # cross-hair
        # self.vLine = pg.InfiniteLine(angle=90, movable=False)
        # self.hLine = pg.InfiniteLine(angle=0, movable=False)

        b-=1
        c = TOOLBAR_WIDTH//2

        # turn on draw mode
        self.SCheckBox = QCheckBox('single stroke')
        self.SCheckBox.setStyleSheet(checkstyle(COLORS[0]))
        self.SCheckBox.setFont(self.medfont)
        self.SCheckBox.toggled.connect(self.autosave_on)
        self.l0.addWidget(self.SCheckBox, b,c,1,TOOLBAR_WIDTH)

        b+=1
        # turn on crosshairs
        self.CHCheckBox = QCheckBox('cross-hairs')
        self.CHCheckBox.setStyleSheet(self.checkstyle)
        self.CHCheckBox.setFont(self.medfont)
        self.CHCheckBox.toggled.connect(self.cross_hairs)
        self.l0.addWidget(self.CHCheckBox, b,c,1,TOOLBAR_WIDTH)
        
        
        
        # b+=1
        # # send to server
        # self.ServerButton = QPushButton(' send manual seg. to server')
        # self.ServerButton.clicked.connect(lambda: save_server(self))
        # self.l0.addWidget(self.ServerButton, b,0,1,8)
        # self.ServerButton.setEnabled(False)
        # self.ServerButton.setStyleSheet(self.styleInactive)
        # self.ServerButton.setFont(self.boldfont)


        
        
        b+=2
        label = QLabel('Segmentation:')
        label.setStyleSheet('color: {}'.format(COLORS[0]))
        label.setFont(self.boldfont)
        self.l0.addWidget(label, b,0,1,TOOLBAR_WIDTH)

#### The segmentation section is where a lot of rearrangement happened 

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
        self.Dimension.currentIndexChanged.connect(self.brush_choose)
        self.Dimension.setStyleSheet(self.dropdowns())
        self.Dimension.setFont(self.medfont)
        self.Dimension.setFixedWidth(WIDTH_3)
        self.l0.addWidget(self.Dimension, b, c,1, TOOLBAR_WIDTH-c)
        
        # CELL DIAMETER text field
        b+=1
        self.diameter = 30
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
        self.affinity = QCheckBox('affinity graph reconstruction')
        self.affinity.setStyleSheet(self.checkstyle)
        self.affinity.setFont(self.medfont)
        self.affinity.setChecked(False)
        self.affinity.setToolTip('sets whether or not to use affinity graph mask reconstruction')
        self.affinity.toggled.connect(self.toggle_affinity)
        self.l0.addWidget(self.affinity, b,0,1,2)

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
        
        # b+=1
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


        # label = QLabel(' Progress:')
        # label.setStyleSheet(label_style)
        # label.setFont(self.medfont)
        # label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        # self.l0.addWidget(label, b,4,1,2)
        
        # b+=2
        # line = QHLine()
        # line.setStyleSheet(self.linestyle)
        # self.l0.addWidget(line, b,0,1,8)

####### Below is the cellpose2.0 arrangement
        
        # # use GPU
        # self.useGPU = QCheckBox('use GPU')
        # self.useGPU.setStyleSheet(self.checkstyle)
        # self.useGPU.setFont(self.medfont)
        # self.useGPU.setToolTip('if you have specially installed the <i>cuda</i> version of torch, then you can activate this')
        # self.check_gpu()
        # self.l0.addWidget(self.useGPU, b,5,1,4)

        # b+=1
        # self.diameter = 30
        # label = QLabel('cell diameter (pixels) (click ENTER):')
        # label.setStyleSheet(label_style)
        # label.setFont(self.medfont)
        # label.setToolTip('you can manually enter the approximate diameter for your cells, \nor press “calibrate” to let the model estimate it. \nThe size is represented by a disk at the bottom of the view window \n(can turn this disk off by unchecking “scale disk on”)')
        # self.l0.addWidget(label, b, 0,1,9)
        # self.Diameter = QLineEdit()
        # self.Diameter.setToolTip('you can manually enter the approximate diameter for your cells, \nor press “calibrate” to let the model estimate it. \nThe size is represented by a disk at the bottom of the view window \n(can turn this disk off by unchecking “scale disk on”)')
        # self.Diameter.setText(str(self.diameter))
        # self.Diameter.setFont(self.medfont)
        # self.Diameter.returnPressed.connect(self.compute_scale)
        # self.Diameter.setFixedWidth(50)
        # b+=1
        # self.l0.addWidget(self.Diameter, b,0,1,5)

        # # recompute model
        # self.SizeButton = QPushButton('  calibrate')
        # self.SizeButton.clicked.connect(self.calibrate_size)
        # self.l0.addWidget(self.SizeButton, b,5,1,4)
        # self.SizeButton.setEnabled(False)
        # self.SizeButton.setStyleSheet(self.styleInactive)
        # self.SizeButton.setFont(self.boldfont)

        # ### fast mode
        # #self.NetAvg = QComboBox()
        # #self.NetAvg.addItems(['average 4 nets', 'run 1 net', '+ turn off resample (fast)'])
        # #self.NetAvg.setFont(self.medfont)
        # #self.NetAvg.setToolTip('average 4 different fit networks (default); run 1 network (faster); or run 1 net + turn off resample (fast)')
        # #self.l0.addWidget(self.NetAvg, b,5,1,4)

        
        # b+=1
        # # choose channel
        # self.ChannelChoose = [QComboBox(), QComboBox()]
        # self.ChannelChoose[0].addItems(['0: gray', '1: red', '2: green','3: blue'])
        # self.ChannelChoose[1].addItems(['0: none', '1: red', '2: green', '3: blue'])
        # cstr = ['chan to segment:', 'chan2 (optional):']
        # for i in range(2):
        #     #self.ChannelChoose[i].setFixedWidth(70)
        #     self.ChannelChoose[i].setStyleSheet(self.dropdowns)
        #     self.ChannelChoose[i].setFont(self.medfont)
        #     label = QLabel(cstr[i])
        #     label.setStyleSheet(label_style)
        #     label.setFont(self.medfont)
        #     if i==0:
        #         label.setToolTip('this is the channel in which the cytoplasm or nuclei exist that you want to segment')
        #         self.ChannelChoose[i].setToolTip('this is the channel in which the cytoplasm or nuclei exist that you want to segment')
        #     else:
        #         label.setToolTip('if <em>cytoplasm</em> model is chosen, and you also have a nuclear channel, then choose the nuclear channel for this option')
        #         self.ChannelChoose[i].setToolTip('if <em>cytoplasm</em> model is chosen, and you also have a nuclear channel, then choose the nuclear channel for this option')
        #     self.l0.addWidget(label, b,0,1,5)
        #     self.l0.addWidget(self.ChannelChoose[i], b,5,1,4)
        #     b+=1

        # # post-hoc paramater tuning

        # b+=1
        # label = QLabel('flow_threshold:')
        # label.setToolTip('threshold on flow error to accept a mask (set higher to get more cells, e.g. in range from (0.1, 3.0), OR set to 0.0 to turn off so no cells discarded);\n press enter to recompute if model already run')
        # label.setStyleSheet(label_style)
        # label.setFont(self.medfont)
        # self.l0.addWidget(label, b, 0,1,5)
        # self.flow_threshold = QLineEdit()
        # self.flow_threshold.setText('0.4')
        # self.flow_threshold.returnPressed.connect(self.run_mask_reconstruction)
        # self.flow_threshold.setFixedWidth(70)
        # self.l0.addWidget(self.flow_threshold, b,5,1,4)

        # b+=1
        # label = QLabel('cellprob_threshold:')
        # label.setToolTip('threshold on cellprob output to seed cell masks (set lower to include more pixels or higher to include fewer, e.g. in range from (-6, 6)); \n press enter to recompute if model already run')
        # label.setStyleSheet(label_style)
        # label.setFont(self.medfont)
        # self.l0.addWidget(label, b, 0,1,5)
        # self.cellprob_threshold = QLineEdit()
        # self.cellprob_threshold.setText('0.0')
        # self.cellprob_threshold.returnPressed.connect(self.run_mask_reconstruction)
        # self.cellprob_threshold.setFixedWidth(70)
        # self.l0.addWidget(self.cellprob_threshold, b,5,1,4)

        # b+=1
        # label = QLabel('stitch_threshold:')
        # label.setToolTip('for 3D volumes, turn on stitch_threshold to stitch masks across planes instead of running cellpose in 3D (see docs for details)')
        # label.setStyleSheet(label_style)
        # label.setFont(self.medfont)
        # self.l0.addWidget(label, b, 0,1,5)
        # self.stitch_threshold = QLineEdit()
        # self.stitch_threshold.setText('0.0')
        # #self.cellprob_threshold.returnPressed.connect(self.run_mask_reconstruction)
        # self.stitch_threshold.setFixedWidth(70)
        # self.l0.addWidget(self.stitch_threshold, b,5,1,4)

        # b+=1
        # self.GB = QGroupBox('model zoo')
        # self.GB.setStyleSheet("QGroupBox { border: 1px solid white; color:white; padding: 10px 0px;}")
        # self.GBg = QGridLayout()
        # self.GB.setLayout(self.GBg)

        # # compute segmentation with general models
        # self.net_text = ['cyto','nuclei','tissuenet','livecell', 'cyto2']
        # nett = ['cellpose cyto model', 
        #         'cellpose nuclei model',
        #         'tissuenet cell model',
        #         'livecell model',
        #         'cellpose cyto2 model']
        # self.StyleButtons = []
        # for j in range(len(self.net_text)):
        #     self.StyleButtons.append(guiparts.ModelButton(self, self.net_text[j], self.net_text[j]))
        #     self.GBg.addWidget(self.StyleButtons[-1], 0,2*j,1,2)
        #     if j < 4:
        #         self.StyleButtons[-1].setFixedWidth(45)
        #     else:
        #         self.StyleButtons[-1].setFixedWidth(35)
        #     self.StyleButtons[-1].setToolTip(nett[j])

        # # compute segmentation with style model
        # self.net_text.extend(['CP', 'CPx', 'TN1', 'TN2', 'TN3', #'TN-p','TN-gi','TN-i',
        #                  'LC1', 'LC2', 'LC3', 'LC4', #'LC-g','LC-e','LC-r','LC-n',
        #                 ])
        # nett = ['cellpose cyto fluorescent', 'cellpose other', 'tissuenet 1', 'tissuenet 2', 'tissuenet 3',
        #         'livecell A172 + SKOV3', 'livecell various', 'livecell BV2 + SkBr3', 'livecell SHSY5Y']
        # for j in range(9):
        #     self.StyleButtons.append(guiparts.ModelButton(self, self.net_text[j+5], self.net_text[j+5]))
        #     self.GBg.addWidget(self.StyleButtons[-1], 1,j,1,2)
        #     self.StyleButtons[-1].setFixedWidth(22)
        #     self.StyleButtons[-1].setToolTip(nett[j])

        # self.StyleToModel = QPushButton(' compute style and run suggested model')
        # self.StyleToModel.setStyleSheet(self.styleInactive)
        # self.StyleToModel.clicked.connect(self.suggest_model)
        # self.StyleToModel.setToolTip(' uses general cp2 model to compute style and runs suggested model based on style')
        # self.StyleToModel.setFont(self.smallfont)
        # self.GBg.addWidget(self.StyleToModel, 2,0,1,10)

        # self.l0.addWidget(self.GB, b, 0, 2, 9)

        # b+=2
        # self.CB = QGroupBox('custom models')
        # self.CB.setStyleSheet("QGroupBox { border: 1px solid white; color:white; padding: 10px 0px;}")
        # self.CBg = QGridLayout()
        # self.CB.setLayout(self.CBg)
        # tipstr = 'add or train your own models in the "Models" file menu and choose model here'
        # self.CB.setToolTip(tipstr)
        
        # # choose models
        # self.ModelChoose = QComboBox()
        # if len(self.model_strings) > 0:
        #     current_index = 0
        #     self.ModelChoose.addItems(['select custom model'])
        #     self.ModelChoose.addItems(self.model_strings)
        # else:
        #     self.ModelChoose.addItems(['select custom model'])
        #     current_index = 0
        # self.ModelChoose.setFixedWidth(180)
        # self.ModelChoose.setStyleSheet(self.dropdowns)
        # self.ModelChoose.setFont(self.medfont)
        # self.ModelChoose.setCurrentIndex(current_index)
        # self.ModelChoose.activated.connect(self.model_choose)
        
        # self.CBg.addWidget(self.ModelChoose, 0,0,1,7)

        # # compute segmentation w/ custom model
        # self.ModelButton = QPushButton(u'run model')
        # self.ModelButton.clicked.connect(self.compute_model)
        # self.CBg.addWidget(self.ModelButton, 0,7,2,2)
        # self.ModelButton.setEnabled(False)
        # self.ModelButton.setStyleSheet(self.styleInactive)

        # self.l0.addWidget(self.CB, b, 0, 1, 9)
        
        # b+=1
        # self.progress = QProgressBar(self)
        # self.progress.setStyleSheet('color: gray;')
        # self.l0.addWidget(self.progress, b,0,1,5)

        self.roi_count = QLabel('0 ROIs')
        # self.roi_count.setStyleSheet('color: white;')
        self.roi_count.setFont(self.boldfont_button)
        self.roi_count.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
        w = TOOLBAR_WIDTH//2+1
        self.l0.addWidget(self.roi_count, b, w,1,TOOLBAR_WIDTH-w)
        
        
        
        
        self.progress = QProgressBar(self)
#         self.progress.setStyleSheet('''QProgressBar {
#                                     border-style: solid;
#                                     border-color: #565656;

#                                     border-width: 1px;
#                                     text-align: center;
#                                 }

#                                 ''')
        # self.progress.setStyleSheet(self.styleInactive)
        # self.progrss.text('Progress')
        self.progress.setValue(0)
        self.l0.addWidget(self.progress, b,0,1,w)
        
#######

        # b+=2
        # line = QHLine()
        # line.setStyleSheet(self.linestyle)
        # self.l0.addWidget(line, b,0,1,8)

        
       

        # b+=1
        # # add z position underneath
        # self.currentZ = 0
        # label = QLabel('Z:')
        # label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        # label.setStyleSheet(label_style)
        # self.l0.addWidget(label, b, 4,1,2)
        # self.zpos = QLineEdit()
        # self.zpos.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        # self.zpos.setText(str(self.currentZ))
        # self.zpos.returnPressed.connect(self.update_ztext)
        # self.zpos.setFixedWidth(60)
        # self.l0.addWidget(self.zpos, b, 6,1,3)
        
        # # scale toggle
        # self.scale_on = True
        # self.ScaleOn = QCheckBox('scale disk on')
        # self.ScaleOn.setFont(self.medfont)
        # self.ScaleOn.setStyleSheet('color: rgb(150,50,150);')
        # self.ScaleOn.setChecked(True)
        # self.ScaleOn.setToolTip('see current diameter as red disk at bottom')
        # self.ScaleOn.toggled.connect(self.toggle_scale)
        # self.l0.addWidget(self.ScaleOn, b,0,1,4)

        # add scrollbar underneath
        self.scroll = QScrollBar(QtCore.Qt.Horizontal)
        # self.scroll.setMaximum(10)
        self.scroll.valueChanged.connect(self.move_in_Z)
        self.l0.addWidget(self.scroll, b,TOOLBAR_WIDTH+1,1,3*b)
        
        # self.l0.addWidget(QLabel(''), b, 0,1,TOOLBAR_WIDTH)        

        
        return b

    
    def dropdowns(self,width=WIDTH_0):
        return ''.join(['QComboBox QAbstractItemView {',
                        # 'background-color: #303030;',
                        # 'selection-color: white; ',
                        'min-width: {};'.format(width),
                        '}'])

    def keyPressEvent(self, event):
        if self.loaded:
            #self.p0.setMouseEnabled(x=True, y=True)
            if (event.modifiers() != QtCore.Qt.ControlModifier and
                event.modifiers() != QtCore.Qt.ShiftModifier and
                event.modifiers() != QtCore.Qt.AltModifier) and not self.in_stroke:
                updated = False
                if len(self.current_point_set) > 0:
                    if event.key() == QtCore.Qt.Key_Return:
                        self.add_set()
                    if self.NZ>1:
                        if event.key() == QtCore.Qt.Key_Left:
                            self.currentZ = max(0,self.currentZ-1)
                            self.scroll.setValue(self.currentZ)
                            updated = True
                        elif event.key() == QtCore.Qt.Key_Right:
                            self.currentZ = min(self.NZ-1, self.currentZ+1)
                            self.scroll.setValue(self.currentZ)
                            updated = True
                else:
                    # toggle masks with X or M
                    if event.key() == QtCore.Qt.Key_X or event.key() == QtCore.Qt.Key_M:
                        self.MCheckBox.toggle()
                    
                    #toggle outlines with Z or O
                    if event.key() == QtCore.Qt.Key_Z or event.key() == QtCore.Qt.Key_O:
                        self.OCheckBox.toggle()
                    
                    # toggle ncolor with C or N
                    if event.key() == QtCore.Qt.Key_C or event.key() == QtCore.Qt.Key_N:
                        self.NCCheckBox.toggle()  

                    # if event.key() == QtCore.Qt.Key_Left:
                    #     if self.NZ==1:
                    #         self.get_prev_image()
                    #     else:
                    #         self.currentZ = max(0,self.currentZ-1)
                    #         self.scroll.setValue(self.currentZ)
                    #         updated = True
                    # elif event.key() == QtCore.Qt.Key_Right:
                    #     if self.NZ==1:
                    #         self.get_next_image()
                    #     else:
                    #         self.currentZ = min(self.NZ-1, self.currentZ+1)
                    #         self.scroll.setValue(self.currentZ)
                    #         updated = True
                    elif event.key() == QtCore.Qt.Key_A:
                        if self.NZ==1:
                            self.get_prev_image()
                        else:
                            self.currentZ = max(0,self.currentZ-1)
                            self.scroll.setValue(self.currentZ)
                            updated = True
                    elif event.key() == QtCore.Qt.Key_D:
                        if self.NZ==1:
                            self.get_next_image()
                        else:
                            self.currentZ = min(self.NZ-1, self.currentZ+1)
                            self.scroll.setValue(self.currentZ)
                            updated = True

                    # elif event.key() == QtCore.Qt.Key_PageDown:
                    #     self.view = (self.view+1)%(len(self.RGBChoose.bstr))
                    #     self.RGBChoose.button(self.view).setChecked(True)
                    # elif event.key() == QtCore.Qt.Key_PageUp:
                    #     self.view = (self.view-1)%(len(self.RGBChoose.bstr))
                    #     self.RGBChoose.button(self.view).setChecked(True)

                # can change background or stroke size if cell not finished
                if event.key() == QtCore.Qt.Key_Up or event.key() == QtCore.Qt.Key_W:
                    self.color = (self.color-1)%(6)
                    self.RGBDropDown.setCurrentIndex(self.color)
                elif event.key() == QtCore.Qt.Key_Down or event.key() == QtCore.Qt.Key_S:
                    self.color = (self.color+1)%(6)
                    self.RGBDropDown.setCurrentIndex(self.color)
                elif event.key() == QtCore.Qt.Key_R:
                    if self.color!=1:
                        self.color = 1
                    else:
                        self.color = 0
                    self.RGBDropDown.setCurrentIndex(self.color)
                elif event.key() == QtCore.Qt.Key_G:
                    if self.color!=2:
                        self.color = 2
                    else:
                        self.color = 0
                    self.RGBDropDown.setCurrentIndex(self.color)
                elif event.key() == QtCore.Qt.Key_B:
                    if self.color!=3:
                        self.color = 3
                    else:
                        self.color = 0
                    self.RGBDropDown.setCurrentIndex(self.color)
                elif (event.key() == QtCore.Qt.Key_Comma or
                        event.key() == QtCore.Qt.Key_Period):
                    count = self.BrushChoose.count()
                    gci = self.BrushChoose.currentIndex()
                    if event.key() == QtCore.Qt.Key_Comma:
                        gci = max(0, gci-1)
                    else:
                        gci = min(count-1, gci+1)
                    self.BrushChoose.setCurrentIndex(gci)
                    self.brush_choose()
                    
                # if not updated: This appears not to be necessary 
                #     self.update_plot()
                
                
                elif event.modifiers() == QtCore.Qt.ControlModifier:
                    if event.key() == QtCore.Qt.Key_Z:
                        self.undo_action()
                    if event.key() == QtCore.Qt.Key_0:
                        self.clear_all()
        if event.key() == QtCore.Qt.Key_Minus or event.key() == QtCore.Qt.Key_Equal:
            self.p0.keyPressEvent(event)

    def check_gpu(self, use_torch=True):
        # also decide whether or not to use torch
        self.torch = use_torch
        self.useGPU.setChecked(False)
        self.useGPU.setEnabled(False)    
        if self.torch and core.use_gpu(use_torch=True):
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
            print(f'GUI_INFO: selected model {self.ModelChoose.currentText()}, loading now')
            self.initialize_model()
            self.diameter = self.model.diam_labels
            self.Diameter.setText('%0.2f'%self.diameter)
            print(f'GUI_INFO: diameter set to {self.diameter: 0.2f} (but can be changed)')

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
            self.remcell.setEnabled(True)
            self.undo.setEnabled(True)
        else:
            self.ClearButton.setEnabled(False)
            self.remcell.setEnabled(False)
            self.undo.setEnabled(False)

    def remove_action(self):
        if self.selected>0:
            self.remove_cell(self.selected)

    def undo_action(self):
        if (len(self.strokes) > 0 and
            self.strokes[-1][0][0]==self.currentZ):
            self.remove_stroke()
        else:
            # remove previous cell
            if self.ncells> 0:
                self.remove_cell(self.ncells)

    def undo_remove_action(self):
        self.undo_remove_cell()

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
        io._masks_to_gui(self, self.cellpix, outlines=self.outpix, format_labels=True)
        self.redraw_masks(masks=self.masksOn, outlines=self.outlinesOn)
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
            
    def make_viewbox(self):
        self.p0 = guiparts.ViewBoxNoRightDrag(
            parent=self,
            lockAspect=True,
            # name="plot1",
            # border=[100, 100, 100],
            invertY=True
        )
        
        
        self.p0.setCursor(QtCore.Qt.CrossCursor)
        self.brush_size=1
        self.win.addItem(self.p0, 0, 0, rowspan=1, colspan=1)
        self.p0.setMenuEnabled(False)
        self.p0.setMouseEnabled(x=True, y=True)
        self.img = pg.ImageItem(viewbox=self.p0, parent=self,levels=(0,255))
        self.img.autoDownsample = False
        
        # self.hist = pg.HistogramLUTItem(image=self.img,orientation='horizontal',gradientPosition='bottom')
        self.hist = guiparts.HistLUT(image=self.img,orientation='horizontal',gradientPosition='bottom')

        self.opacity_effect = QtWidgets.QGraphicsOpacityEffect()
        self.hist.setGraphicsEffect(self.opacity_effect)

        # self.set_hist_colors() #called elsewhere. no need
        # print(self.hist.__dict__)
        # self.win.addItem(self.hist,col=0,row=2)
        self.win.addItem(self.hist,col=0,row=1)


        self.layer = guiparts.ImageDraw(viewbox=self.p0, parent=self)
        self.scale = pg.ImageItem(viewbox=self.p0, parent=self,levels=(0,255))

        self.Ly,self.Lx = 512,512
        
        self.p0.scene().contextMenuItem = self.p0
        self.p0.addItem(self.img)
        self.p0.addItem(self.layer)
        self.p0.addItem(self.scale)

        
        # policy = QtWidgets.QSizePolicy()
        # policy.setRetainSizeWhenHidden(True)
        # self.hist.setSizePolicy(policy)

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
        color = '#efefef' if self.darkmode else '#888888'
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
        self.BrushChoose.setCurrentIndex(1)
        self.SCheckBox.setChecked(True)
        self.SCheckBox.setEnabled(False)
        self.restore_masks = 0
        self.states = [None for i in range(len(self.default_cmaps))] 

        # -- zero out image stack -- #
        self.opacity = 128 # how opaque masks should be
        self.outcolor = [200,200,255,200]
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
        self.ismanual = np.zeros(0, 'bool')
        self.accent = self.palette().brush(QPalette.ColorRole.Highlight).color()
        self.update_plot()
        self.progress.setValue(0)
        self.orthobtn.setChecked(False)
        self.filename = []
        self.loaded = False
        self.recompute_masks = False


    def brush_choose(self):
        if self.BrushChoose.currentIndex() > 0:
            self.brush_size = (self.BrushChoose.currentIndex()-1)*2 + 1
        else:
            self.brush_size = 0
        if self.loaded:
            self.layer.setDrawKernel(kernel_size=self.brush_size)
            self.update_layer()

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
        self.prev_selected = 0
        self.selected = 0
        self.layerz = np.zeros((self.Ly,self.Lx,4), np.uint8)
        self.cellpix = np.zeros((self.NZ,self.Ly,self.Lx), np.uint32)
        self.outpix = np.zeros((self.NZ,self.Ly,self.Lx), np.uint32)
        self.cellcolors = np.array([255,255,255])[np.newaxis,:]
        self.ncells = 0
        self.toggle_removals()
        self.update_layer()

    def select_cell(self, idx):
        self.prev_selected = self.selected
        self.selected = idx
        if self.selected > 0:
            z = self.currentZ
            self.layerz[self.cellpix[z]==idx] = np.array([255,255,255,self.opacity])
            self.update_layer()

    def unselect_cell(self):
        if self.selected > 0:
            idx = self.selected
            if idx < self.ncells+1:
                z = self.currentZ
                self.layerz[self.cellpix[z]==idx] = np.append(self.cellcolors[idx], self.opacity)
                if self.outlinesOn:
                    self.layerz[self.outpix[z]==idx] = np.array(self.outcolor).astype(np.uint8)
                    #[0,0,0,self.opacity])
                self.update_layer()
        self.selected = 0

    def remove_cell(self, idx):
        # remove from manual array
        self.selected = 0
        if self.NZ > 1:
            zextent = ((self.cellpix==idx).sum(axis=(1,2)) > 0).nonzero()[0]
        else:
            zextent = [0]
        for z in zextent:
            cp = self.cellpix[z]==idx
            op = self.outpix[z]==idx
            # remove from self.cellpix and self.outpix
            self.cellpix[z, cp] = 0
            self.outpix[z, op] = 0    
            if z==self.currentZ:
                # remove from mask layer
                self.layerz[cp] = np.array([0,0,0,0])

        # reduce other pixels by -1
        self.cellpix[self.cellpix>idx] -= 1
        self.outpix[self.outpix>idx] -= 1
        
        if self.NZ==1:
            self.removed_cell = [self.ismanual[idx-1], self.cellcolors[idx], np.nonzero(cp), np.nonzero(op)]
            self.redo.setEnabled(True)
            ar, ac = self.removed_cell[2]
            d = datetime.datetime.now()        
            self.track_changes.append([d.strftime("%m/%d/%Y, %H:%M:%S"), 'removed mask', [ar,ac]])
        # remove cell from lists
        self.ismanual = np.delete(self.ismanual, idx-1)
        self.cellcolors = np.delete(self.cellcolors, [idx], axis=0)
        del self.zdraw[idx-1]
        self.ncells -= 1
        print('GUI_INFO: removed cell %d'%(idx-1))
        
        self.update_layer()
        if self.ncells==0:
            self.ClearButton.setEnabled(False)
        if self.NZ==1:
            io._save_sets(self)

    def merge_cells(self, idx):
        self.prev_selected = self.selected
        self.selected = idx
        if self.selected != self.prev_selected:
            for z in range(self.NZ):
                ar0, ac0 = np.nonzero(self.cellpix[z]==self.prev_selected)
                ar1, ac1 = np.nonzero(self.cellpix[z]==self.selected)
                touching = np.logical_and((ar0[:,np.newaxis] - ar1)<3,
                                            (ac0[:,np.newaxis] - ac1)<3).sum()
                ar = np.hstack((ar0, ar1))
                ac = np.hstack((ac0, ac1))
                vr0, vc0 = np.nonzero(self.outpix[z]==self.prev_selected)
                vr1, vc1 = np.nonzero(self.outpix[z]==self.selected)
                self.outpix[z, vr0, vc0] = 0    
                self.outpix[z, vr1, vc1] = 0    
                if touching > 0:
                    mask = np.zeros((np.ptp(ar)+4, np.ptp(ac)+4), np.uint8)
                    mask[ar-ar.min()+2, ac-ac.min()+2] = 1
                    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    pvc, pvr = contours[-2][0].squeeze().T            
                    vr, vc = pvr + ar.min() - 2, pvc + ac.min() - 2
                    
                else:
                    vr = np.hstack((vr0, vr1))
                    vc = np.hstack((vc0, vc1))
                color = self.cellcolors[self.prev_selected]
                self.draw_mask(z, ar, ac, vr, vc, color, idx=self.prev_selected)
            self.remove_cell(self.selected)
            print('GUI_INFO: merged two cells')
            self.update_layer()
            io._save_sets(self)
            self.undo.setEnabled(False)      
            self.redo.setEnabled(False)    

    def undo_remove_cell(self):
        if len(self.removed_cell) > 0:
            z = 0
            ar, ac = self.removed_cell[2]
            vr, vc = self.removed_cell[3]
            color = self.removed_cell[1]
            self.draw_mask(z, ar, ac, vr, vc, color)
            self.toggle_mask_ops()
            self.cellcolors = np.append(self.cellcolors, color[np.newaxis,:], axis=0)
            self.ncells+=1
            self.ismanual = np.append(self.ismanual, self.removed_cell[0])
            self.zdraw.append([])
            print('>>> added back removed cell')
            self.update_layer()
            io._save_sets(self)
            self.removed_cell = []
            self.redo.setEnabled(False)


    def remove_stroke(self, delete_points=True, stroke_ind=-1):
        #self.current_stroke = get_unique_points(self.current_stroke)
        stroke = np.array(self.strokes[stroke_ind])
        cZ = self.currentZ
        inZ = stroke[0,0]==cZ
        if inZ:
            outpix = self.outpix[cZ, stroke[:,1],stroke[:,2]]>0
            self.layerz[stroke[~outpix,1],stroke[~outpix,2]] = np.array([0,0,0,0])
            cellpix = self.cellpix[cZ, stroke[:,1], stroke[:,2]]
            ccol = self.cellcolors.copy()
            if self.selected > 0:
                ccol[self.selected] = np.array([255,255,255])
            col2mask = ccol[cellpix]
            if self.masksOn:
                col2mask = np.concatenate((col2mask, self.opacity*(cellpix[:,np.newaxis]>0)), axis=-1)
            else:
                col2mask = np.concatenate((col2mask, 0*(cellpix[:,np.newaxis]>0)), axis=-1)
            self.layerz[stroke[:,1], stroke[:,2], :] = col2mask
            if self.outlinesOn:
                self.layerz[stroke[outpix,1],stroke[outpix,2]] = np.array(self.outcolor)
            if delete_points:
                self.current_point_set = self.current_point_set[:-1*(stroke[:,-1]==1).sum()]
            self.update_layer()
            
        del self.strokes[stroke_ind]

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
            print('ERROR: zposition is not a number')
        self.currentZ = max(0, min(self.NZ-1, zpos))
        self.zpos.setText(str(self.currentZ))
        self.scroll.setValue(self.currentZ)

    def update_plot(self):
        self.Ly, self.Lx, _ = self.stack[self.currentZ].shape
        
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
        if state is None: #should adda button to reset state to none and update plot
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

    def update_layer(self):
        self.draw_layer()
        # if (self.masksOn or self.outlinesOn) and self.view==0:
        self.layer.setImage(self.layerz, autoLevels=False)
            # self.layer.setImage(self.layerz[self.currentZ], autoLevels=False)
            
        self.update_roi_count()
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
            

    def add_set(self):
        if len(self.current_point_set) > 0:
            self.current_point_set = np.array(self.current_point_set)
            while len(self.strokes) > 0:
                self.remove_stroke(delete_points=False)
            if len(self.current_point_set) > 8:
                color = self.colormap[self.ncells,:3]
                median = self.add_mask(points=self.current_point_set, color=color)
                if median is not None:
                    self.removed_cell = []
                    self.toggle_mask_ops()
                    self.cellcolors = np.append(self.cellcolors, color[np.newaxis,:], axis=0)
                    self.ncells+=1
                    self.ismanual = np.append(self.ismanual, True)
                    if self.NZ==1:
                        # only save after each cell if single image
                        io._save_sets(self)
            self.current_stroke = []
            self.strokes = []
            self.current_point_set = []
            self.update_layer()

    def add_mask(self, points=None, color=(100,200,50)):
        # loop over z values
        median = []
        if points.shape[1] < 3:
            points = np.concatenate((np.zeros((points.shape[0],1), "int32"), points), axis=1)

        zdraw = np.unique(points[:,0])
        zrange = np.arange(zdraw.min(), zdraw.max()+1, 1, int)
        zmin = zdraw.min()
        pix = np.zeros((2,0), "uint16")
        mall = np.zeros((len(zrange), self.Ly, self.Lx), "bool")
        k=0
        for z in zdraw:
            iz = points[:,0] == z
            vr = points[iz,1]
            vc = points[iz,2]
            # get points inside drawn points
            mask = np.zeros((np.ptp(vr)+4, np.ptp(vc)+4), np.uint8)
            pts = np.stack((vc-vc.min()+2,vr-vr.min()+2), axis=-1)[:,np.newaxis,:]
            mask = cv2.fillPoly(mask, [pts], (255,0,0))
            ar, ac = np.nonzero(mask)
            ar, ac = ar+vr.min()-2, ac+vc.min()-2
            # get dense outline
            contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            pvc, pvr = contours[-2][0].squeeze().T            
            vr, vc = pvr + vr.min() - 2, pvc + vc.min() - 2
            # concatenate all points
            ar, ac = np.hstack((np.vstack((vr, vc)), np.vstack((ar, ac))))
            # if these pixels are overlapping with another cell, reassign them
            ioverlap = self.cellpix[z][ar, ac] > 0
            if (~ioverlap).sum() < 8:
                print('ERROR: cell too small without overlaps, not drawn')
                return None
            elif ioverlap.sum() > 0:
                ar, ac = ar[~ioverlap], ac[~ioverlap]
                # compute outline of new mask
                mask = np.zeros((np.ptp(ar)+4, np.ptp(ac)+4), np.uint8)
                mask[ar-ar.min()+2, ac-ac.min()+2] = 1
                contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                pvc, pvr = contours[-2][0].squeeze().T            
                vr, vc = pvr + ar.min() - 2, pvc + ac.min() - 2
            self.draw_mask(z, ar, ac, vr, vc, color)

            median.append(np.array([np.median(ar), np.median(ac)]))
            mall[z-zmin, ar, ac] = True
            pix = np.append(pix, np.vstack((ar, ac)), axis=-1)

        mall = mall[:, pix[0].min():pix[0].max()+1, pix[1].min():pix[1].max()+1].astype(np.float32)
        ymin, xmin = pix[0].min(), pix[1].min()
        if len(zdraw) > 1:
            mall, zfill = interpZ(mall, zdraw - zmin)
            for z in zfill:
                mask = mall[z].copy()
                ar, ac = np.nonzero(mask)
                ioverlap = self.cellpix[z+zmin][ar+ymin, ac+xmin] > 0
                if (~ioverlap).sum() < 5:
                    print('WARNING: stroke on plane %d not included due to overlaps'%z)
                elif ioverlap.sum() > 0:
                    mask[ar[ioverlap], ac[ioverlap]] = 0
                    ar, ac = ar[~ioverlap], ac[~ioverlap]
                # compute outline of mask
                outlines = masks_to_outlines(mask)
                vr, vc = np.nonzero(outlines)
                vr, vc = vr+ymin, vc+xmin
                ar, ac = ar+ymin, ac+xmin
                self.draw_mask(z+zmin, ar, ac, vr, vc, color)
        self.zdraw.append(zdraw)
        if self.NZ==1:
            d = datetime.datetime.now()
            self.track_changes.append([d.strftime("%m/%d/%Y, %H:%M:%S"), 'added mask', [ar,ac]])
        return median

    def draw_mask(self, z, ar, ac, vr, vc, color, idx=None):
        ''' draw single mask using outlines and area '''
        if idx is None:
            idx = self.ncells+1
        self.cellpix[z, vr, vc] = idx
        self.cellpix[z, ar, ac] = idx
        self.outpix[z, vr, vc] = idx
        self.layerz[ar, ac, :3] = color
        if self.masksOn:
            self.layerz[ar, ac, -1] = self.opacity
        if self.outlinesOn:
            self.layerz[vr, vc] = np.array(self.outcolor)


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

    def redraw_masks(self, masks=True, outlines=True, draw=True):
        self.draw_layer()

    def draw_masks(self):
        self.draw_layer()

    def draw_layer(self):
        if self.masksOn and self.view==0: #disable masks for network outputs
            self.layerz = np.zeros((self.Ly,self.Lx,4), np.uint8)
            self.layerz[...,:3] = self.cellcolors[self.cellpix[self.currentZ],:]
            self.layerz[...,3] = self.opacity * (self.cellpix[self.currentZ]>0).astype(np.uint8)
            if self.selected>0:
                self.layerz[self.cellpix[self.currentZ]==self.selected] = np.array([255,255,255,self.opacity])
            cZ = self.currentZ
            stroke_z = np.array([s[0][0] for s in self.strokes])
            inZ = np.nonzero(stroke_z == cZ)[0]
            if len(inZ) > 0:
                for i in inZ:
                    stroke = np.array(self.strokes[i])
                    self.layerz[stroke[:,1], stroke[:,2]] = np.array([255,0,255,100])
        else:
            self.layerz[...,3] = 0

        if self.outlinesOn:
            self.layerz[self.outpix[self.currentZ]>0] = np.array(self.outcolor).astype(np.uint8)

    # def compute_saturation(self):
    #     # compute percentiles from stack
    #     self.saturation = []
    #     print('GUI_INFO: auto-adjust enabled, computing saturation levels')
    #     if self.NZ>10:
    #         iterator = trange(self.NZ)
    #     else:
    #         iterator = range(self.NZ)
    #     for n in iterator:
    #         self.saturation.append([np.percentile(self.stack[n].astype(np.float32),1),
    #                                 np.percentile(self.stack[n].astype(np.float32),99)])

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
            print('ERROR: cannot train model on 3D data')
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
            print('GUI_INFO: training cancelled')

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
        
        print('GUI_INFO: name of new model:' + self.training_params['model_name'])
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
        
    # The CP2 GUI uses a text box instead of a slider... I prefer the slider  
#     def run_mask_reconstruction(self):
#         if self.recompute_masks:
#             flow_threshold, cellprob_threshold = self.get_thresholds()
#             if flow_threshold is None:
#                 logger.info('computing masks with cell prob=%0.3f, no flow error threshold'%
#                         (cellprob_threshold))
#             else:
#                 logger.info('computing masks with cell prob=%0.3f, flow error threshold=%0.3f'%
#                         (cellprob_threshold, flow_threshold))
#             maski = dynamics.compute_masks(self.flows[-1][:-1], 
#                                             self.flows[-1][-1],
#                                             p=self.flows[3].copy(),
#                                             cellprob_threshold=cellprob_threshold,
#                                             flow_threshold=flow_threshold,
#                                             resize=self.cellpix.shape[-2:])[0]
            
#             self.masksOn = True
#             self.MCheckBox.setChecked(True)
#             # self.outlinesOn = True #should not turn outlines back on by default; masks make sense though 
#             # self.OCheckBox.setChecked(True)
#             if maski.ndim<3:
#                 maski = maski[np.newaxis,...]
#             logger.info('%d cells found'%(len(np.unique(maski)[1:])))
#             io._masks_to_gui(self, maski, outlines=None)
#             self.show()

    def run_mask_reconstruction(self):
        # use_omni = 'omni' in self.current_model
        
        # needed to be replaced with recompute_masks
        # rerun = False
        have_enough_px = self.probslider.value() < self.cellprob # slider moves down
        # if self.cellprob != self.probslider.value():
        #     rerun = True
        #     self.cellprob = self.probslider.value()
            
        # if self.threshold != self.threshslider.value():
        #     rerun = True
        #     self.threshold = self.threshslider.value()

        # print('bbbbbb', self.recompute_masks,self.cellprob, self.threshold)
        if not self.recompute_masks:
            return
        
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
            
            # must recompute flows if we add pixels, because p does not contain them
            # an alternate approach would be to compute p for the lowest allowed threshold
            # and then never ecompute (the threshold prodces a mask that selects from existing trajectories, see get_masks)
            p = self.flows[-2].copy() if have_enough_px else None 
            
            dP = self.flows[-1][:-self.model.dim]
            dist = self.flows[-1][self.model.dim]
            bd = self.flows[-1][self.model.dim+1]
            # print('flow debug',self.model.dim,p.shape,dP.shape,dist.shape,bd.shape)
            maski = omnipose.core.compute_masks(dP=dP, 
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
                                                affinity_seg=self.affinity.isChecked(),
                                                omni=omni)[0]

        
        self.masksOn = True
        self.MCheckBox.setChecked(True)
        # self.outlinesOn = True #should not turn outlines back on by default; masks make sense though 
        # self.OCheckBox.setChecked(True)
        if maski.ndim<3:
            maski = maski[np.newaxis,...]
        logger.info('%d cells found'%(len(np.unique(maski)[1:])))
        io._masks_to_gui(self, maski, outlines=None) # replace this to show boundary emphasized masks
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
        for i in range(len(self.net_text)):
            self.StyleButtons[i].setStyleSheet(self.styleUnpressed)
        self.StyleButtons[ind].setStyleSheet(self.stylePressed)
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
            
            # print('heredebug',self.stack.shape,data.shape, channels)
            
            ### will either have to put in edge cases for worm etc or just generalize model loading to respect what is there 
            try:
                omni_model = 'omni' in self.current_model
                bacterial = 'bact' in self.current_model
                if omni_model or bacterial:
                    self.NetAvg.setCurrentIndex(1) #one run net
                if bacterial:
                    self.diameter = 0.
                    self.Diameter.setText('%0.1f'%self.diameter)

                # allow omni to be togged manually or forced by model
                if OMNI_INSTALLED:
                    if omni_model:
                        print('GUI_INFO: turning on Omnipose mask recontruction version for Omnipose models (see menu)')
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
                print
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
                                               affinity_seg=self.affinity.isChecked(),
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

            io._masks_to_gui(self, masks, outlines=None)
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
        self.SCheckBox.setEnabled(True)
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

        
        
