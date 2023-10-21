import setuptools
from setuptools import setup
from setuptools_scm import get_version
import os

# Get the absolute path to the directory containing setup.py
base_dir = os.path.abspath(os.path.dirname(__file__))

# Construct the absolute path to the cellpose-omni subpackage
cellpose_omni_path = os.path.join(base_dir, 'cellpose-omni')

install_deps = ['numpy>=1.22.4', 'scipy', 'numba', 
                'edt','scikit-image','ncolor>=1.2.1',
                'scikit-learn','torch>=1.10',
                'mahotas>=1.4.13',
                'mgen','matplotlib',
                'peakdetect','igraph',
                'torchvf','tqdm',
                # 'cellpose-omni',
                'cellpose-omni @ file://{}'.format(cellpose_omni_path) # local version
                # 'file://{}#egg=cellpose-omni'.format(cellpose_omni_path) # local version in editable mode

                # 'torchvf@git+https://github.com/kevinjohncutler/torchvf.git'
                ]

# install_deps.append('cellpose-omni{} @ file://{}'.format('[gui]' if 'gui' in sys.argv else '',cellpose_omni_path))

doc_deps = ['sphinx-autobuild',
            'sphinx_automodapi',
            'sphinx_copybutton',
            'sphinx_design','furo','myst_nb']
    
with open("README.md", "r") as fh:
    long_description = fh.read() 
    
    setup(
    name="omnipose",
    version=get_version(),
    author="Kevin Cutler",
    author_email="kevinjohncutler@outlook.com",
    description="cell segmentation algorithm improving on the Cellpose framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kevinjohncutler/omnipose",
    setup_requires=[
        'pytest-runner',
        'setuptools_scm',
    ],
    packages=setuptools.find_packages(),
    # use_scm_version=True,
    # install_requires = install_deps+cp_deps,
    install_requires = install_deps,
    extras_require={
        "gui": ["cellpose-omni[gui] @ file://{}".format(cellpose_omni_path)]
    },
    # dependency_links=["file://{}".format(cellpose_omni_path)],
    tests_require=[
      'pytest'
    ],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    # package_data={
    #     'omnipose': ['cellpose-omni/*'],
    # },
    entry_points = {
        'console_scripts': [
          'omnipose = omnipose.__main__:main']
    },
)
