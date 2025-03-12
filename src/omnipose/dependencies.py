# install_deps = ['numpy>=1.22.4,<=1.26.4', # will need to wait a bit for cythonized packages to catch up to numpy 2.0
install_deps = ['numba>=0.61.0', # let numba control numpy version 
                # 'numpy>=1.22.4,<2', # maybe it is safe now? Nope, as of January 2025
                'scipy', 
                'edt',
                'scikit-image',
                'ncolor>=1.2.1',
                'scikit-learn',
                'torch>=1.10',
                'torchvision', # redundant from torchvf 
                'mahotas>=1.4.13',
                'mgen',
                'matplotlib',
                'ipywidgets', # technically could factor out to a notebook dep list 
                'networkit',
                'torchvf',
                'tqdm', 
                'natsort', 
                'aicsimageio', # should make this optional, include czi dep
                'numexpr',
                'torch_optimizer', # for RADAM, now supported directly in pytorch though 
                'tifffile', # might be dependency of aicsimageio
                'fastremap', # not sure how I missed this one 
                'cmap', 
                'dbscan', # almost 2x faster than sklearn dbscan!
                'pyinstrument'
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
        'pyopengl'
        ]

distributed_deps = [
        'dask',
        'dask_image',
        'scikit-learn',
        ]

