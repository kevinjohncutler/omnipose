from setuptools import setup, find_packages
import sys, os
sys.path.append(os.path.dirname(__file__))

from dependencies import install_deps, gui_deps, distributed_deps

with open('docs/requirements.txt') as f:
    doc_deps = [line.strip() for line in f if line.strip() and not line.startswith('#') and '-e .' not in line]

with open("README.rst", "r") as fh:
    long_description = fh.read() 

setup(
    name="omnipose",
    use_scm_version=True,
    author="Kevin Cutler",
    author_email="kevinjohncutler@outlook.com",
    description="cell segmentation algorithm improving on the Cellpose framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kevinjohncutler/omnipose",
    packages=find_packages(include=['omnipose', 'cellpose_omni']),
    install_requires = install_deps,
    extras_require = {
      'gui': gui_deps,
      'docs': doc_deps,
      'all': doc_deps + gui_deps + distributed_deps,
    },
    tests_require=[
      'pytest'
    ],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    entry_points = {
        'console_scripts': [
          'omnipose = omnipose.__main__:main']
    },
    py_modules=['dependencies'],

)
