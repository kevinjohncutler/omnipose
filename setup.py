import setuptools
from setuptools import setup

install_deps = ['numpy>=1.22.4', 'scipy', 'numba', 
                'edt','scikit-image','ncolor',
                #hdbscan, should I include this?
                'mgen','scikit-learn',
                'mahotas@git+https://github.com/luispedro/mahotas#egg=mahotas', # 1.4.13 binary not compatible with some versions of numpy
                'cellpose[all]@git+https://github.com/kevinjohncutler/cellpose#egg=cellpose[all]',
                'mgen@git+https://github.com/kevinjohncutler/mgen#egg=mgen',] # my version just removes stuff for pyinstalelr to work 

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
    install_requires = install_deps,
    tests_require=[
      'pytest'
    ],
    include_package_data=True,
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    )
)
