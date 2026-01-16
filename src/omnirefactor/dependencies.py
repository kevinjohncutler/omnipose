# install_deps = ['numpy>=1.22.4,<=1.26.4', # will need to wait a bit for cythonized packages to catch up to numpy 2.0
install_deps = ['numba>=0.61.0', # let numba control numpy version 
                # 'numpy>=1.22.4,<2', # maybe it is safe now? Nope, as of January 2025
                'numpy', # let numba control numpy version - v2 now supported by all dependencies! July 2025
                'scipy', 
                'edt',
                'scikit-image',
                'ncolor>=1.4.3',
                'scikit-learn',
                'torch>=1.10',
                'torchvision', # redundant from torchvf 
                'mahotas>=1.4.13', # not sure I use this anymore
                'mgen',
                'matplotlib',
                'ipywidgets', # technically could factor out to a notebook dep list 
                'networkit',
                'torchvf',
                'tqdm', 
                'natsort', 
                'aicsimageio', # should make this optional, include czi dep
                'numexpr',
                'torch_optimizer', # for RADAM, now supported directly in pytorch though... 
                'tifffile', # might be dependency of aicsimageio, so not needed explicitly 
                'fastremap',
                'cmap', 
                'colour-science', # called "colour" when importing, but "colour-science" on pypi
                'dbscan', # almost 2x faster than sklearn dbscan!
                'pyinstrument',
                'imagecodecs', # should be able to get rid of a lot of cv2 
                'opencv-python-headless', # headless version of opencv, no GUI stuff
                # 'opencv-contrib-python-headless', # headless version of opencv, no GUI stuff
                'opt_einsum', # for faster einsum, not sure if needed long term
                
                'dask',
                'dask-image',
                'scikit-learn',
                ]

# notes: Numpy 2 is close, networkit might be the last dependency needed to upgrade 
# Now it works! Tested july 2025 with 2.2.6, 0.61.2

gui_deps = [
        'pyqtgraph>=0.12.4', 
        'PyQt6',
        # 'google-cloud-storage',
        'omnipose-theme', # my fork of pyqtdarktheme        
        'superqt',
        'darkdetect',
        'qtawesome',
        'pyopengl',
        'imageio'
        ]

distributed_deps = [] # moved to main deps

