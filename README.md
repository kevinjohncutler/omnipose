<img src="https://github.com/kevinjohncutler/omnipose/blob/main/logo3.png?raw=true" width="200" title="bacteria" alt="bacteria" align="right" vspace = "0">
<img src="https://github.com/kevinjohncutler/omnipose/blob/main/logo.png?raw=true" width="200" title="omnipose" alt="omnipose" align="center" vspace = "0">

[![Downloads](https://static.pepy.tech/personalized-badge/omnipose?period=total&units=international_system&left_color=grey&right_color=green&left_text=Downloads)](https://pepy.tech/project/omnipose)
[![PyPI version](https://badge.fury.io/py/omnipose.svg)](https://badge.fury.io/py/omnipose)

[cp]: https://github.com/MouseLand/cellpose

Omnipose is a general image segmentation tool that builds on [Cellpose][cp] in a number of ways described in our [paper](https://www.nature.com/articles/s41592-022-01639-4). It works for both 2D and 3D images and on any imaging modality or cell shape, so long as you train it on representative images. We have several pre-trained models for:
* **bacterial phase contrast**: trained on a diverse range of bacterial species and morphologies. 
* **bacterial fluorescence**: trained on the subset of the phase data that had a membrane or cytosol tag. 
* ***C. elegans***: trained on a couple OpenWorm videos and the [BBBC010](https://bbbc.broadinstitute.org/BBBC010) alive/dead assay. We are working on expanding this significantly with the help of other labs contributing ground-truth data. 
* **cyto2**: trained on user data submitted through the Cellpose GUI. Very diverse data, but not necessarily the best quality. This model can be a good starting point for users making their own ground-truth datasets. 

## Try out Omnipose online

[colab]: https://colab.research.google.com/github/HenriquesLab/ZeroCostDL4Mic/blob/master/Colab_notebooks/Beta%20notebooks/Cellpose_2D_ZeroCostDL4Mic.ipynb
[ZeroCostDL4Mic]: https://github.com/HenriquesLab/ZeroCostDL4Mic/wiki

New users can check out the [ZeroCostDL4Mic][ZeroCostDL4Mic] Cellpose notebook on [Google Colab][colab] to try out our original release of Omnipose. We need to make sure this gets updated to the most recent version of Omnipose with advanced 3D features and more built-in models. 

## Use the GUI

Launch the Omnipose-optimized version of the Cellpose GUI from terminal: `omnipose`. Version 0.4.0 and onward will *not* install the GUI dependencies by default. When you first run the GUI command, you will be prompted to install the GUI dependencies. On Ubuntu 2022.04 (and possibly earlier), we found it necessary to run the following to install a missing system package: 

```
sudo apt install libxcb-xinerama0
```
Our version of the GUI gives easy access to the parameters you need to run Omnipose in large batches via CLI or Jupyter notebooks. The [ncolor](https://github.com/kevinjohncutler/ncolor) label representation is now default and can be toggled off for saving masks in standard format. 

Standalone versions of this GUI for Windows, macOS, and Linux are available on the [OSF repository](https://osf.io/xmury/).

## How to install Omnipose

1. Install an [Anaconda](https://www.anaconda.com/download/) distribution of Python. Note you might need to use an anaconda prompt if you did not add anaconda to the path. Alternatives like miniconda also work just as well. 
2. Open an anaconda prompt / command prompt with `conda` for **python 3** in the path.
3. To create a new environment for CPU only, run
    ```
    conda create -n omnipose 'python==3.10.12' pytorch
    
    ```
    For users with NVIDIA GPUs, add these additional arguments:
    ```
    torchvision pytorch-cuda=11.8 -c pytorch -c nvidia 

    ```
    [See GPU support](#gpu-support) for more details. Python 3.10 is not a strict requirement; see [Python compatibility](#python-compatibility) for more about choosing your python version.
    
4. To activate this new environment, run 
    ```
    conda activate omnipose
    ```
5. To install the latest PyPi release of Omnipose, run
    ``` 
    pip install omnipose
    ``` 
    or, for the most up-to-date development version,
    ```
    git clone https://github.com/kevinjohncutler/omnipose.git
    cd omnipose
    python install.py gui
    ```
    (Omit `gui` if you do not want the gui installed). 
    
> :warning: **Version conflicts**: if you previously installed Omnipose, please run 
> ```
> pip uninstall cellpose_omni && pip cache remove cellpose_omni
> ```
> to prevent version conflicts. See [project structure](#project-structure) for more details. 
    
### Python compatibility 
We have tested Omnipose extensively on Python version 3.8.5 and have encountered issues on some lower versions. Versions up to 3.10.11 have been confirmed compatible, but we have encountered bugs with the GUI dependencies on 3.11+. For those users with system or global pyenv python3 installations, check your python version by running `python -V` before making your conda environment and choose a different version. That way, there is no crosstalk between pip-installed packages inside and outside your environment. So if you have 3.x.y installed via pyenv etc., install your environment with 3.x.z instead. 

### Pyenv versus Conda

Pyenv also works great for creating an environment for installing Omnipose (and it also works a lot better for installing Napari alongside it, in my experience). Simply set your global version anywhere from 3.8.5-3.10.11 and run `pip install omnipose`. I've had no problems with GPU compatibility with this method on Linux, as pip collects all the required packages. Conda is much more reproducible, but often finicky. You can use pyenv on Windows and macOS too, but you will need a conda environment for Apple Silicon GPU support (PyPi still lacks many package versions built for Apple Silicon). 

### GPU support 

Omnipose runs on CPU on macOS, Windows, and Linux. PyTorch has historically only supported NVIDIA GPUs, but has more more recently begun supporting Apple Silicon GPUs. It looks AMD support may be avaiable these days (ROCm), but I have not tested that out. Windows and Linux installs are straightforward:

Your PyTorch version (>=1.6) needs to be compatible with your NVIDIA driver. Older cards may not be supported by the latest drivers and thus not supported by the latest PyTorch version. See the official documentation on installing both the [most recent](https://pytorch.org/get-started/locally/) and [previous](https://pytorch.org/get-started/previous-versions/) combinations of CUDA and PyTorch to suit your needs. Accordingly, you can get started with CUDA 11.8 by making the following environment:
```
conda create -n omnipose 'python==3.10.12' pytorch torchvision pytorch-cuda=11.8 \
-c pytorch -c nvidia 

```
Note that the official PyTorch command includes torchaudio, but that is not needed for Omnipose. (*torchvision appears to be necessary these days*). If you are on older drivers, you can get started with an older version of CUDA, *e.g.* 10.2:
```
conda create -n omnipose pytorch=1.8.2 cudatoolkit=10.2 -c pytorch-lts
```
For Apple Silicon, download [omnipose_mac_environment.yml](omnipose_mac_environment.yml) and install the environment:

```
conda env create -f <path_to_environment_file>
conda activate omnipose
```

You may edit this yml to change the name or python version etc. For more notes on Apple Silicon development, see [this thread](https://github.com/kevinjohncutler/omnipose/issues/14). On all systems, remember that you may need to use ipykernel to use the omnipose environment in a notebook. 

## How to use Omnipose
I have a few Jupyter notebooks in the [docs/examples](docs/examples/) directory that demonstrate how to use built-in models. You can also find all the scripts I used for generating our figures in the [scripts](scripts/) directory. These cover specific settings for all of the images found in our paper. 

To use Omnipose on bacterial cells, use `model_type=bact_omni`. For other cell types, try `model_type=cyto2_omni`. You can also choose Cellpose models with `omni=True` to engage the Omnipose mask reconstruction algorithm to alleviate over-segmentation. 


## How to train Omnipose
Training is best done on CLI. I trained the `bact_phase_omni` model using the following command, and you can train custom Omnipose models similarly:

```
omnipose --train --use_gpu --dir <bacterial dataset directory> --mask_filter _masks \
         --n_epochs 4000 --pretrained_model None --learning_rate 0.1 --diameter 0 \
         --batch_size 16 --RAdam --img_filter _img --nclasses 3
```

On bacterial phase contrast data, I found that Cellpose does not benefit much from more than 500 epochs but Omnipose continues to improve until around 4000 epochs. Omnipose outperforms Cellpose at 500 epochs but is significantly better at 4000. You can use `--save_every <n>` and `--save_each` to store intermediate model training states to explore this behavior. 


## 3D Omnipose

To train a 3D model on image volumes, specify the dimension argument: `--dim 3`. You may run out of VRAM on your GPU. In that case, you can specify a smaller crop size, *e.g.*, `--tyx 50,50,50`. The command I used in the paper on the *Arabidopsis thaliana* lateral root primordia dataset was:
```
omnipose --use_gpu --train --dir <path> --mask_filter _masks \
         --n_epochs 4000 --pretrained_model None --learning_rate 0.1 --save_every 50 \
         --save_each  --verbose --look_one_level_down --all_channels --dim 3 \
         --RAdam --batch_size 4 --diameter 0 --nclasses 3
```

To evaluate Omnipose models on 3D data, see the [examples](docs/examples/). If you run out of GPU memory, consider (a) evaluating on CPU or (b) using `tile=True`. 


## Known limitations
Cell size remains the only practical limitation of Omnipose. On the low end, cells need to be at least 3 pixels wide in each dimension. On the high end, 60px appears to work well, with 150px being too large. The current workaround is to first downscale your images so that cells are within an appropriate size range (3-60px). This can be done automatically during training with `--diameter <X>`. The mean cell diameter `D` is calculated from the ground truth masks and images are rescaled by `X/D`. 


## Project structure, feature requests, and issues 
Omnipose is built on [Cellpose][cp], and functionally that means Cellpose actually imports Omnipose to replace many of its operations with the Omnipose versions with `omni=True`. Omnipose was first packaged into the Cellpose repo before I began making too many ND-generalizations (full rewrites) for the authors to maintain. Thus was birthed my `cellpose_omni` fork, which I published to PyPi separately from Omnipose for some time. I later decided that maintaining two packages for one project was overcomplicated for me and users (especially for installations from the repo), so the latest version of `cellpose_omni` now lives here. `cellpose_omni` still gets installed as its own subpackage when you install Omnipose. If you have issues migrating to the new version, make sure to `pip uninstall omnipose cellpose_omni` before re-installing Omnipose. The `install.py` script simply runs `pip install -e .{extras}` in the `omnipose` and `cellpose` directories. 

If you encounter bugs with Omnipose, you can check the [main Cellpose repo][cp] for related issues and also post them here. I do my best to keep up with with bug fixes and features from the main branch, but it helps me out a lot if users bring them to my attention. If there are any features or pull requests in Cellpose that you want to see in Omnipose ASAP, please let me know. 


## Building the GUI app

PyInstaller can be used to compile Omnipose into a standalone app. The limitation is that the build process itself needs to run within the OS on which the app will be run. We plan to release app versions for macOS 12.3, Windows 10, and Ubuntu 20.04, which should also work on newer versions of each OS. I will periodically update these apps for the public, but we will also post notes below to guide others in compiling the code:

1. Start with a fresh conda environment with only the dependencies that Omnipose and pyinstaller need. 
2. `cd` into the pyinstaller directory and run
    ``` 
    pyinstaller --clean --noconfirm --onefile omni.py --collect-all pyqtgraph
    ``` 
    This will make a `build` and `dist` folder. `--onefile` makes an executable that opens up a terminal window. This is important because the GUI still outputs information there, especially with the debug box checked. This bare-bones command generates the omni.spec file that can be further edited. At this point, this minimal setup produces very large executibles (>300MB) depending on the OS, but they are functional.
3. numpy seems to be the limiting factor preventing us from making universal2 executibles. This means that Intel (osx_64) and Apple Silicon (osx_arm64) apps need to be frozen separately on their respective platforms. The former works just the same as Windows and Ubuntu. The latter was a bit of a nightmare, as I had to ensure that all possible dependencies of Omnipose *and* Cellpose were manually installed from miniforge into a clean conda environment to get the osx_arm64 builds. I then installed Omnipose, which only needed to pip install the few other packages like ncolor and mgen that were not already installed via conda. I also needed to upgrade my fork of Cellpose, where the GUI lives, to PyQt6 (previously PyQt5). An environment.yaml is sorely needed to make this process easier. However, on osx_arm64 I found it necessary to additionally include a `--collect all skimage`:
    ``` 
    pyinstaller --clean --noconfirm --onefile omni.py --collect-all pyqtgraph --collect-all skimage
    ``` 

4. On macOS, there is a `NSRequiresAquaSystemAppearance` variable that needs to be set to `False` so that the app respects the system theme (no white title bar if you are in dark mode). I made this change in omni_mac.spec. To build off the spec file, run 
    ```
    pyinstaller --noconfirm omni_mac.spec
    ``` 
    
Some more notes: 
- the mgen dependency had some version declarations that are incompatible with pyinstaller. Install my fork of mgen prior to building the app. 

pyinstaller --clean --noconfirm --onefile omni.py --collect-all pyqtgraph --collect-all skimage --collect-all torch

## Licensing
See `LICENSE.txt` for details. This license does not affect anyone using Omnipose for noncommercial applications. 
