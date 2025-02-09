import signal, sys, os, pathlib, warnings, datetime, time
import inspect, importlib, pkgutil

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

from PyQt6.QtCore import QTimer
import importlib




os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '1'

from scipy.stats import mode
# from scipy.ndimage import gaussian_filter

# from . import guiparts, menus, io
from cellpose_omni.gui import guiparts, menus, io
from .. import models, dynamics
from ..utils import download_url_to_file, masks_to_outlines, diameters 
from ..io import get_image_files, imsave, imread, check_dir #OMNI_INSTALLED
from ..transforms import resize_image #fixed import
from ..plot import disk
from omnipose.utils import normalize99, to_8_bit


OMNI_INSTALLED = 1
from .guiutils import checkstyle, get_unique_points, avg3d, interpZ

from . import logger


ALLOWED_THEMES = ['light','dark']



import darkdetect
import qdarktheme
import qtawesome as qta


# no more matplotlib just for colormaps
from cmap import Colormap

from . import MainWindowModules as submodules

from . import PRELOAD_IMAGE, ICON_PATH

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
        
        
        # Dict mapping { module_name: last_mtime }
        self.module_mtimes = {}

        # Discover & load all submodules
        self.modules = self.load_all_submodules()
        self.patch_all_submodules()

        # Start a timer to check for changes every second
        self.timer_id = self.startTimer(1000)

        
        
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

        self.make_main_widget()

        self.imask = 0

        b = self.make_buttons()
        # self.cwidget.setStyleSheet('border: 1px; border-radius: 10px')
        

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


    def load_all_submodules(self):
        """
        Dynamically imports every .py module in submodules/ 
        and returns a dict of {mod_name: module_object}.
        """
        loaded_modules = {}
        # submodules.__path__ = the path of the submodules/ folder
        for mod_info in pkgutil.iter_modules(submodules.__path__):
            mod_name = mod_info.name
            full_name = submodules.__name__ + "." + mod_name
            # Import the module
            mod = importlib.import_module(full_name)
            loaded_modules[mod_name] = mod

            # Track its last modified time
            mod_path = os.path.join(os.path.dirname(submodules.__file__), mod_name + ".py")
            self.module_mtimes[mod_name] = self.get_mtime(mod_path)
        return loaded_modules
        
    # def load_all_submodules(self):
    #     """
    #     Dynamically imports every .py module in submodules/
    #     and returns a dict of {mod_name: module_object}.
    #     """
    #     loaded_modules = {}
        
    #     # submodules.__path__ is always defined (list of dirs),
    #     # even if submodules is a namespace package.
    #     parent_dir = next(iter(submodules.__path__))
        
    #     import pkgutil
    #     for mod_info in pkgutil.iter_modules(submodules.__path__):
    #         mod_name = mod_info.name
    #         full_name = submodules.__name__ + "." + mod_name
            
    #         # Import the submodule
    #         mod = importlib.import_module(full_name)
    #         loaded_modules[mod_name] = mod

    #         # Build the path like "parent_dir / mod_name.py"
    #         mod_path = os.path.join(parent_dir, mod_name + ".py")
    #         self.module_mtimes[mod_name] = self.get_mtime(mod_path)
        
    #     return loaded_modules

    def timerEvent(self, event):
        """
        Called every second; checks if any submodule .py changed on disk.
        If so, reload it and patch again.
        """
        for mod_name, mod in self.modules.items():
            mod_path = os.path.join(os.path.dirname(submodules.__file__), mod_name + ".py")
            new_mtime = self.get_mtime(mod_path)
            if new_mtime != self.module_mtimes[mod_name]:
                # The file changed => reload
                print(f"🔄 Reloading submodule '{mod_name}'...")
                new_mod = importlib.reload(mod)
                self.modules[mod_name] = new_mod
                self.module_mtimes[mod_name] = new_mtime
                # Re-patch after reloading
                self.patch_submodule(new_mod)

    def patch_all_submodules(self):
        """Call patch_submodule() for each loaded module."""
        for mod in self.modules.values():
            self.patch_submodule(mod)

    def patch_submodule(self, mod):
        """
        For each top-level function in a submodule, 
        bind it as a method on MainWindow.
        """
        for name, obj in inspect.getmembers(mod, inspect.isfunction):
            bound_method = obj.__get__(self, type(self))
            setattr(self, name, bound_method)
            # print(f"Patched {name}() from '{mod.__name__}' -> self.{name}")

    def call_patched_method(self, func_name):
        """Call a hot-reloaded method by name."""
        if hasattr(self, func_name):
            getattr(self, func_name)()
        else:
            print(f"No method '{func_name}' found on MainWindow")

    def get_mtime(self, filepath):
        """Return last-modified time of a file, or 0 if missing."""
        return os.path.getmtime(filepath) if os.path.exists(filepath) else 0
 
    
        
        
 
        

 





    # def dragEnterEvent(self, event):
    #     if event.mimeData().hasUrls():
    #         event.accept()
    #     else:
    #         event.ignore()

    
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

    # this is where we need to change init depending on whether or not we have a size model
    # or which size model to use... 
    # this should really be updated to allow for custom size models to be used, too
    # I guess most doing that will not be using the GUI, but still an important feature 




        
# prevents gui from running under import 
if __name__ == "__main__":
    run()