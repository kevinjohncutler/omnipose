from setuptools import setup, find_packages
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from omnipose.dependencies import install_deps, gui_deps, distributed_deps

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="omnipose",
    version="2.0.0",
    author="Kevin Cutler",
    author_email="kevinjohncutler@outlook.com",
    description="Omnipose: cell segmentation algorithm and GUI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kevinjohncutler/omnipose",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        # Ship the dock/window icon (and the larger logo) so the GUI launcher
        # can find them when installed from a wheel.
        "omnipose.gui": ["*.png"],
    },
    include_package_data=True,
    install_requires=install_deps,
    extras_require={
        "gui": gui_deps,
        "all": gui_deps + distributed_deps,
        "dev": [
            "pytest",
            "pytest-cov",
            "coverage",
            "genbadge[coverage]",
        ],
    },
    tests_require=[
        "pytest",
    ],
    entry_points={
        "console_scripts": [
            "omnipose = omnipose.__main__:main",
            "omnipose-loss-server = omnipose.cli.loss_server:main",
            "omnipose-sweep-report = omnipose.cli.sweep_report:main",
        ],
        "ocdkit.plugins": [
            "omnipose = omnipose.gui:plugin",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
