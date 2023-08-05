import setuptools
from setuptools import setup

install_deps = ['numpy>=1.22.4', 'scipy', 'numba', 
                'edt','scikit-image','ncolor>=1.2.1',
                'scikit-learn','torch>=1.10',
                'mahotas>=1.4.13',
                'mgen','matplotlib',
                'peakdetect','igraph',
                'torchvf',

                # 'torchvf@git+https://github.com/kevinjohncutler/torchvf.git'
                ]

doc_deps = ['sphinx-autobuild',
            'sphinx_automodapi',
            'sphinx_copybutton',
            'sphinx_design','furo','myst_nb']

import os

# if os.getenv('NO_GUI'):
#     extra = 'omni'
# else:
#     extra = 'all'
# extra = 'omni'
# cp_ver = '0.9.1'
# cp_deps = ['cellpose-omni[{}]>={}'.format(extra,cp_ver),]
    
with open("README.md", "r") as fh:
    long_description = fh.read() 
    
setup(
    name="omnipose",
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
    packages=setuptools.find_packages(exclude=("omnipose.scripts",)),
    use_scm_version=True,
    # install_requires = install_deps+cp_deps,
    install_requires = install_deps,

    tests_require=[
      'pytest'
    ],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
