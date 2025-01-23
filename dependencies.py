# install_deps = ['numpy>=1.22.4,<=1.26.4', # will need to wait a bit for cythonized packages to catch up to numpy 2.0
install_deps = ['numpy>=1.22.4,<2', # maybe it is safe now? Nope, as of January 2025
                'scipy', 
                'numba', 
                'edt',
                'scikit-image',
                'ncolor>=1.2.1',
                'scikit-learn',
                'torch>=1.10',
                'torchvision',
                'mahotas>=1.4.13',
                'mgen',
                'matplotlib',
                'networkit',
                'torchvf',
                'tqdm', 
                'natsort', 
                'aicsimageio', # should make this optional, include czi dep
                'numexpr',
                'torch_optimizer', # for RADAM
                'tifffile', # might be dependency of aicsimageio
                'fastremap', # not sure how I missed this one 
                'cmap' 
                ]

# notes: Numpy 2 is close, networkit might be the last dependency needed to upgrade 

gui_deps = [
        'pyqtgraph>=0.12.4', 
        'PyQt6.sip', 
        'PyQt6',
        # 'google-cloud-storage',
        'omnipose-theme',
        'superqt',
        'darkdetect',
        'qtawesome',
        ]

distributed_deps = [
        'dask',
        'dask_image',
        'scikit-learn',
        ]

