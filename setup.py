from setuptools import setup, find_packages
import os


def _read_requirements(path):
    """Parse a pip requirements.txt — strip comments and blank lines."""
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if line:
                out.append(line)
    return out


_HERE = os.path.dirname(__file__)
install_deps = _read_requirements(os.path.join(_HERE, "requirements.txt"))
# Optional groups installable via ``pip install omnipose[gui]``.  Kept inline
# rather than in separate requirements files — one file is enough.
gui_deps = [
    "imageio",
    "pywebview",
    "fastapi",
    "uvicorn",
    "tensorboard",
]
distributed_deps: list[str] = []

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
