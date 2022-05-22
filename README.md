<img src="https://github.com/kevinjohncutler/omnipose/blob/main/logo3.png?raw=true" width="200" title="bacteria" alt="bacteria" align="right" vspace = "0">
<img src="https://github.com/kevinjohncutler/omnipose/blob/main/logo.png?raw=true" width="200" title="omnipose" alt="omnipose" align="center" vspace = "0">

[![Downloads](https://static.pepy.tech/personalized-badge/omnipose?period=total&units=international_system&left_color=grey&right_color=green&left_text=Downloads)](https://pepy.tech/project/omnipose)
[![PyPI version](https://badge.fury.io/py/omnipose.svg)](https://badge.fury.io/py/omnipose)

[cp]: https://github.com/MouseLand/cellpose

Omnipose is a general image segmentation tool that builds on [Cellpose][cp] in a number of ways described in our [paper](http://biorxiv.org/content/early/2021/11/04/2021.11.03.467199). It works for both 2D and 3D images and on any imaging modality or cell shape, so long as you train it on representative images. We have several pre-trained models for:
* **bacterial phase contrast**: trained on a diverse range of bacterial species and morphologies. 
* **bacterial fluorescence**: trained on the subset of the phase data that had a membrane or cytosol tag. 
* ***C. elegans***: trained on a couple OpenWorm videos and the [BBBC010](https://bbbc.broadinstitute.org/BBBC010) alive/dead assay. We are working on expanding this significantly with the help of other labs contributing ground-truth data. 
* **cyto2**: trained on user data submitted through the Cellpose GUI. Very diverse data, but not necessarily the best quality. This model can be a good starting point for users making their own ground-truth datasets. 


## How to install Omnipose

1. Install an [Anaconda](https://www.anaconda.com/download/) distribution of Python. Note you might need to use an anaconda prompt if you did not add anaconda to the path.
2. Open an anaconda prompt / command prompt with `conda` for **python 3** in the path.
3. To create a new environment, run
```
conda create --name omnipose python=3.8.5
```
4. To activate this new environment, run 
```
conda activate omnipose
```
5. To install Omnipose, run 
```
pip install omnipose
``` 
or 
```
pip install git+https://github.com/kevinjohncutler/omnipose.git
```

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

We have a few Jupyter notebooks in the [examples](examples/) directory that demonstrate how to use built-in models. You can also find all the scripts we used for generating our figures in the [figures](figures/) directory. These cover specific settings for all of the images found in our paper. 

To use Omnipose on bacterial cells, use `model_type=bact_omni`. For other cell types, try `model_type=cyto2_omni`. You can also choose Cellpose models with `omni=True` to engage the Omnipose mask reconstruction algorithm to alleviate over-segmentation. 


## How to train Omnipose
Training is best done on CLI. We trained our `bact_omni` model using the following command, and you can train custom Omnipose models similarly:

```
python -m cellpose --train --use_gpu --dir <bacterial dataset directory> --mask_filter _masks --n_epochs 4000 --pretrained_model None --learning_rate 0.1 --diameter 0 --batch_size 16 --omni --RADAM
```

On our bacterial phase contrast data, we found that while Cellpose does not benefit much from more than 500 epochs, but Omnipose continues to improve until around 4000 epochs. Omnipose outperforms Cellpose at 500 epochs but is significantly better at 4000. You can use `--save_every <n>` and `--save_each` to store intermediate model training states to explore this behavior. 


## 3D Omnipose

To train a 3D model on image volumes, used the dimension argument: `--dim 3`. You may have to choose a smaller crop size for images sent to the GPU. In that case, you can specify a smaller crop size, *e.g.*, `--tyx 50,50,50`. The command we used in the paper on the *Arabidopsis thaliana* lateral root primordia dataset was:
```
python -m cellpose --train --use_gpu --dir ./plantseg/traintest/LateralRootPrimordia/export_small/train --mask_filter _masks --n_epochs 4000 --pretrained_model None  --learning_rate 0.1 --save_every 50 --save_each --omni --verbose --look_one_level_down --all_channels --dim 3 --RAdam --batch_size 4 --diameter 0
```

To evaluate Omnipose models on 3D data, see our [examples](examples/). If you run out of GPU memory, consider (a) evaluating on CPU or (b) using `tile=True`. 


## Known limitations
Cell size remains the only practical limitation of Omnipose. On the low end, cells need to be at least 3 pixels wide in each dimension. On the high end, 60px appears to work well, with 150px being too large. The current workaround is to first downscale your images so that cells are within an appropriate size range (3-60px). This can be done automatically during training with `--diameter <X>`. The mean cell diameter `D` is calculated from the ground truth masks and images are rescaled by `X/D`. 


## Issues and feature requests
As Omnipose is built on [Cellpose][cp], this repo serves mostly to contain new Omnipose-specific functions (like the smooth distance field and the mean cell diameter metric) and our versions of key Cellpose functions (like mask reconstruction). The main Cellpose code base imports these functions and uses them with `omni=True`. This approach was not feasible for our more recent work with ND volume processing, which required extensive rewrites to the filing handling and network architecture that could not be so easily separated from the original code base (and arguably should not, as these changes are the same ideas just expressed in a dimension-agnostic way). For the foreseeable future, our [fork](https://github.com/kevinjohncutler/cellpose) of Cellpose will be the only version compatible with new development of Omnipose after 0.2.1, and it is installed automatically when you install Omnipose. 

This means that if you encounter bugs with Omnipose, you can check the [main Cellpose repo][cp] for related issues and also post them here. Our Cellpose fork will continue to be updated with bug fixes and features from the main branch. If there are any features or pull requests that you want to see in Omnipose ASAP, please let us know. 


## Licensing
See `LICENSE.txt` for details. This license does not affect anyone using Omnipose for noncommercial applications. 
