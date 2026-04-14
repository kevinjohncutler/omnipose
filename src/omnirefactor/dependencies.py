# install_deps = ['numpy>=1.22.4,<=1.26.4', # will need to wait a bit for cythonized packages to catch up to numpy 2.0
install_deps = ['ocdkit[spatial]', # shared utilities (array, gpu, io, measure, morphology, spatial)
                'numba>=0.61.0', # let numba control numpy version
                'numpy', # let numba control numpy version - v2 now supported by all dependencies! July 2025
                'scipy', 
                'edt',
                'scikit-image>=0.26',
                'ncolor>=1.4.3',
                'torch>=1.10',
                'torchvision', # redundant from torchvf 
                'mgen',
                'matplotlib',
                'ipywidgets', # technically could factor out to a notebook dep list 
                'networkit',
                'torchvf', # do we need this anymore? 
                'tqdm', 
                'natsort', 
                'bioio',
                'bioio-czi', # CZI file support plugin
                'numexpr',
                'torch_optimizer', # for RADAM, now supported directly in pytorch though... 
                'tifffile',
                'fastremap',
                'cmap', 
                'dbscan', # almost 2x faster than sklearn dbscan!
                'pyinstrument', # profiling 
                'imagecodecs',
                'opt_einsum', # for faster einsum, not sure if needed long term
                'opensimplex', # ND simplex noise for illumination augmentation
                'dask',
                ]

gui_deps = [
        'imageio',
        'pywebview',
        ]

distributed_deps = [] # moved to main deps
