from setuptools import setup, find_packages
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from omnirefactor.dependencies import install_deps, gui_deps, distributed_deps

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="omnirefactor",
    use_scm_version=True,
    author="Kevin Cutler",
    author_email="kevinjohncutler@outlook.com",
    description="Omnipose refactor package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kevinjohncutler/omnipose",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        "omnirefactor.gui": [
            "web/*",
            "web/*/*",
            "web/*/*/*",
            "web/*/*/*/*",
        ],
    },
    install_requires=install_deps,
    extras_require={
        "gui": gui_deps,
        "all": gui_deps + distributed_deps,
    },
    tests_require=[
        "pytest",
    ],
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "omnirefactor = omnirefactor.__main__:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
