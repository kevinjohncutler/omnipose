import setuptools
from setuptools import setup
from setuptools_scm import get_version

install_deps = ['numpy>=1.22.4', 'scipy', 'natsort',
                'tifffile', 'tqdm', 'numba', 
                'torch>=1.6',
                'opencv-python-headless', # headless not working with pyinstaller?
                'fastremap', 'imagecodecs',
                ]

gui_deps = [
        'pyqtgraph>=0.12.4', 
        'PyQt6.sip', 
        'PyQt6',
        'google-cloud-storage',
        'omnipose-theme',
        # 'PyQtDarkTheme@git+https://github.com/kevinjohncutler/omnipose-theme#egg=PyQtDarkTheme',
        'superqt','colour','darkdetect'
        ]

docs_deps = [
        'sphinx>=3.0',
        'sphinxcontrib-apidoc',
        'sphinx_rtd_theme',
        ]

omni_deps = [
        'scikit-image', 
        'scikit-learn',
        'edt',
        'torch_optimizer', 
        'ncolor>=1.2.1'
        # 'ncolor@git+https://github.com/kevinjohncutler/ncolor#egg=ncolor'
        ]

distributed_deps = [
        'dask',
        'dask_image',
        'scikit-learn',
        ]

# conda install numba numpy tifffile imagecodecs scipy fastremap pyqtgraph
#  pip install opencv-python==4.5.3.56 

try:
    import torch
    a = torch.ones(2, 3)
    version = int(torch.__version__[2])
    if version >= 6:
        install_deps.remove('torch>=1.6')
except:
    pass

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setup(
    name="cellpose-omni",
    version=get_version(root='..', relative_to=__file__),  # use version number from omnipose package
    license="BSD",
    author="Kevin Cutler",
    author_email="kevinjohncutler@outlook.com",
    description="cellpose fork developed for omnipose",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kevinjohncutler/omnipose/tree/main/cellpose-omni",
    setup_requires=[
      'pytest-runner',
      'setuptools_scm',
    ],
    packages=setuptools.find_packages(),
    # use_scm_version=True,
    install_requires = install_deps+omni_deps,
    tests_require=[
      'pytest'
    ],
    extras_require = {
      # 'omni': omni_deps, force omni deps 
      'docs': docs_deps,
      'gui': gui_deps,
      'all': gui_deps + omni_deps,
      'distributed': distributed_deps,
    },
    include_package_data=True,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ),
     entry_points = {
        'console_scripts': [
          'cellpose_omni = cellpose_omni.__main__:main']
     },
)
