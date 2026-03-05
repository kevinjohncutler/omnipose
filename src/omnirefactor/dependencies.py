# install_deps = ['numpy>=1.22.4,<=1.26.4', # will need to wait a bit for cythonized packages to catch up to numpy 2.0
install_deps = ['numba>=0.61.0', # let numba control numpy version 
                'numpy', # let numba control numpy version - v2 now supported by all dependencies! July 2025
                'scipy', 
                'edt',
                'scikit-image>=0.26',
                'ncolor>=1.4.3',
                'scikit-learn',
                'torch>=1.10',
                'torchvision', # redundant from torchvf 
                'mgen',
                'matplotlib',
                'ipywidgets', # technically could factor out to a notebook dep list 
                'networkit',
                'torchvf', # do we need this anymore? 
                'tqdm', 
                'natsort', 
                'aicsimageio', # should make this optional, include czi dep
                'numexpr',
                'torch_optimizer', # for RADAM, now supported directly in pytorch though... 
                'tifffile', # might be dependency of aicsimageio, so not needed explicitly 
                'fastremap',
                'cmap', 
                'dbscan', # almost 2x faster than sklearn dbscan!
                'pyinstrument', # profiling 
                'imagecodecs', # should be able to get rid of a lot of cv2 
                'opencv-python-headless', # headless version of opencv, no GUI stuff
                # 'opencv-contrib-python-headless', # headless version of opencv, no GUI stuff
                'opt_einsum', # for faster einsum, not sure if needed long term
                
                'dask',
                ]

gui_deps = [
        'imageio',
        'pywebview',
        ]

distributed_deps = [] # moved to main deps
