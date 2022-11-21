import setuptools
from setuptools import setup

install_deps = ['numpy>=1.22.4', 'scipy', 'numba', 
                'edt','scikit-image','ncolor',
                'scikit-learn',
                'mahotas>=1.4.13',
                'mgen']

gui_deps = ['cellpose-omni[all]>=0.7.3',]

import os

if os.getenv('NO_GUI'):
    extra = 'omni'
else:
    extra = 'all'

cp_deps = ['cellpose-omni['+extra+']>=0.6.9',]
    
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
    packages=setuptools.find_packages(),
    use_scm_version=True,
    install_requires = install_deps+cp_deps,
    extras_require = {
      'all': gui_deps,
    },
    tests_require=[
      'pytest'
    ],
    include_package_data=True,
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
)
