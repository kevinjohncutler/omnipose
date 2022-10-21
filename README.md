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

Launch our Omnipose-optimized version of the Cellpose GUI from terminal: `python -m omnipose`. The latest version of Omnipose (see GitHub installation) automatically downloads the GUI dependencies of our Cellpose fork. On Ubuntu 2022.04 (and possibly earlier), we found it necessary to run the following to install a missing package: 
```
sudo apt install libxcb-xinerama0
```
Our version of the GUI gives easy access to the parameters you need to run Omnipose in large batches via CLI or Jupyter notebooks. The [ncolor](https://github.com/kevinjohncutler/ncolor) label representation is now default and can be toggled off for saving masks in standard format. 

Standalone versions of this GUI for Windows, macOS, and Linux are coming soon. 

## How to install Omnipose

1. Install an [Anaconda](https://www.anaconda.com/download/) distribution of Python. Note you might need to use an anaconda prompt if you did not add anaconda to the path.
2. Open an anaconda prompt / command prompt with `conda` for **python 3** in the path.
3. To create a new environment, run
    ```
    conda create --name omnipose 'python>=3.8.5'
    ```
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
    pip install git+https://github.com/kevinjohncutler/omnipose.git
    ```

We have tested Omnipose extensively on Python version 3.8.5 and have encountered issues on some lower versions. Versions up to 3.10.4 have been confirmed compatible. Check your python version by running `python -V`. If your version is too low (this happens if your default/base environment python is a lower version), make a new environment and specify the python version, e.g. `conda create --name omnipose python==3.8.5`

### Install without the GUI dependencies 

Set the environment variable `NO_GUI` prior to running the install commands. In linux/macOS, `export NO_GUI=True` (the value here does not matter, `setup.py` just checks for the variable existence). To undo, use `unset NO_GUI`. In Windows, use `set NO_GUI=True` and `set NO_GUI=`. 

### GPU support 

Omnipose runs on CPU on MacOS, Windows, and Linux. PyTorch only supports NVIDIA GPUs, so you can only take advantage of GPU speeds on Linux or Windows. Your PyTorch version (>=1.6) needs to be compatible with your CUDA toolkit version and your NVIDIA driver. See [here](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) for compatibility chart of CUDA and diver versions. Also see the official documentation on installing both the [most recent](https://pytorch.org/get-started/locally/) and [previous](https://pytorch.org/get-started/previous-versions/) combinations of CUDA and PyTorch to suit your needs. Accordingly, you can get started with CUDA 11.3 by making the following environment:
```
conda create -n omnipose pytorch cudatoolkit=11.3 -c pytorch 
```
To get started with CUDA 10.2, instead run:
```
conda create -n omnipose pytorch=1.8.2 cudatoolkit=10.2 -c pytorch-lts
```

## How to use Omnipose
We have a few Jupyter notebooks in the [docs/examples](docs/examples/) directory that demonstrate how to use built-in models. You can also find all the scripts we used for generating our figures in the [scripts](scripts/) directory. These cover specific settings for all of the images found in our paper. 

To use Omnipose on bacterial cells, use `model_type=bact_omni`. For other cell types, try `model_type=cyto2_omni`. You can also choose Cellpose models with `omni=True` to engage the Omnipose mask reconstruction algorithm to alleviate over-segmentation. 


## How to train Omnipose
Training is best done on CLI. We trained our `bact_omni` model using the following command, and you can train custom Omnipose models similarly:

```
python -m omnipose --train --use_gpu --dir <bacterial dataset directory> --mask_filter _masks --n_epochs 4000 --pretrained_model None --learning_rate 0.1 --diameter 0 --batch_size 16 --RAdam --img_filter _img
```

On our bacterial phase contrast data, we found that while Cellpose does not benefit much from more than 500 epochs, but Omnipose continues to improve until around 4000 epochs. Omnipose outperforms Cellpose at 500 epochs but is significantly better at 4000. You can use `--save_every <n>` and `--save_each` to store intermediate model training states to explore this behavior. 


## 3D Omnipose

To train a 3D model on image volumes, specify the dimension argument: `--dim 3`. You may have run out of VRAM on your GPU. In that case, you can specify a smaller crop size, *e.g.*, `--tyx 50,50,50`. The command we used in the paper on the *Arabidopsis thaliana* lateral root primordia dataset was:
```
python -m omnipose --train --use_gpu --dir ./plantseg/traintest/LateralRootPrimordia/export_small/train --mask_filter _masks --n_epochs 4000 --pretrained_model None  --learning_rate 0.1 --save_every 50 --save_each  --verbose --look_one_level_down --all_channels --dim 3 --RAdam --batch_size 4 --diameter 0
```

To evaluate Omnipose models on 3D data, see our [examples](examples/). If you run out of GPU memory, consider (a) evaluating on CPU or (b) using `tile=True`. 


## Known limitations
Cell size remains the only practical limitation of Omnipose. On the low end, cells need to be at least 3 pixels wide in each dimension. On the high end, 60px appears to work well, with 150px being too large. The current workaround is to first downscale your images so that cells are within an appropriate size range (3-60px). This can be done automatically during training with `--diameter <X>`. The mean cell diameter `D` is calculated from the ground truth masks and images are rescaled by `X/D`. 


## Issues and feature requests
As Omnipose is built on [Cellpose][cp], this repo serves mostly to contain new Omnipose-specific functions (like the smooth distance field and the mean cell diameter metric) and our versions of key Cellpose functions (like mask reconstruction). The main Cellpose code base imports these functions and uses them with `omni=True`. This approach was not feasible for our more recent work with ND volume processing, which required extensive rewrites to the filing handling and network architecture that could not be so easily separated from the original code base (and arguably should not, as these changes are the same ideas just expressed in a dimension-agnostic way). For the foreseeable future, our [fork](https://github.com/kevinjohncutler/cellpose) of Cellpose will be the only version compatible with new development of Omnipose after 0.2.1, and it is installed automatically when you install Omnipose. 

This means that if you encounter bugs with Omnipose, you can check the [main Cellpose repo][cp] for related issues and also post them here. Our Cellpose fork will continue to be updated with bug fixes and features from the main branch. If there are any features or pull requests that you want to see in Omnipose ASAP, please let us know. 

## Building the GUI app

PyInstaller can be used to compile Omnipose into a standalone app. The limitation is that the build process itself needs to run within the OS on which the app will be run. We plan to release app versions for macOS 12.3, Windows 10, and Ubuntu 20.04, which should also work on newer versions of each OS. We will periodically update these apps for the public, but we will also post notes below to guide others in compiling the code:

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
- the mgen dependency had some version declarations that are incompatible with pysintaller. Omnipose therefore now depends on my fork that fixes this issue. 

pyinstaller --clean --noconfirm --onefile omni.py --collect-all pyqtgraph --collect-all skimage --collect-all torch

## Licensing
See `LICENSE.txt` for details. This license does not affect anyone using Omnipose for noncommercial applications. 
