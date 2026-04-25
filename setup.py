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
        # Ship the dock/window icon (and the larger logo) so the GUI launcher
        # can find them when installed from a wheel.
        "omnirefactor.gui": ["*.png"],
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
            "omnirefactor = omnirefactor.__main__:main",
            "omnirefactor-loss-server = omnirefactor.cli.loss_server:main",
            "omnirefactor-sweep-report = omnirefactor.cli.sweep_report:main",
        ],
        "ocdkit.plugins": [
            "omnipose = omnirefactor.gui:plugin",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
