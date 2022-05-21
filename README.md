<img src="https://github.com/kevinjohncutler/omnipose/blob/main/logo3.png?raw=true" width="200" title="bacteria" alt="bacteria" align="right" vspace = "0">
<img src="https://github.com/kevinjohncutler/omnipose/blob/main/logo.png?raw=true" width="200" title="omnipose" alt="omnipose" align="center" vspace = "0">

### What is Omnipose
Omnipose is a general image segmentation tool that builds on Cellpose in a number of ways described in our [paper](http://biorxiv.org/content/early/2021/11/04/2021.11.03.467199). It works  for both 2D and 3D images and on any imaging modality or cell shape, so long as you train it on representative images. We have several pre-trained models for:
* **bacterial phase contrast**: trained on a diverse range of bacterial species and morphologies. 
* **bacterial fluorescence**: trained on the subset of the phase data that had a membrane or cytosol tag. 
* **C. elegans**: trained on a couple OpenWorm videos and the [BBBC010](https://bbbc.broadinstitute.org/BBBC010) alive/dead assay. We are working on expanding this significantly with the help of other labs contributing ground-truth data. 
* **cyto2**: trained on user data submitted through the Cellpose GUI. Very diverse data, but not necessarily the best quality. This model can be a good starting point for users making their own ground-truth datasets. 

### How to install Omnipose

1. Install an [Anaconda](https://www.anaconda.com/download/) distribution of Python. Note you might need to use an anaconda prompt if you did not add anaconda to the path.
2. Open an anaconda prompt / command prompt with `conda` for **python 3** in the path.
3. Create a new environment with `conda create --name omnipose python=3.8.5`.
4. To activate this new environment, run `conda activate omnipose`
5. To install Omnipose, run `pip install omnipose` or `pip install git+https://github.com/kevinjohncutler/omnipose.git`

### How to use Omnipose

We have a couple Jupyter notebooks in the [examples](examples/) directory that demonstrate how to use built-in models. You can also find all the scripts we used for generating our figures in the [figures](figures/) directory. These cover specific settings for all of the images found in our paper. 

To use Omnipose on bacterial cells, use `model_type=bact_omni`. For other cell types, try `model_type=cyto2_omni`. You can also choose Cellpose models with `omni=True` to engage the Omnipose mask reconstruction algorithm to alleviate over-segmentation. 


### How to train Omnipose
Training is best done on CLI. We trained our `bact_omni` model using the following command, and you can train custom Omnipose models similarly:

`python -m cellpose --train --use_gpu --dir <bacterial dataset directory> --mask_filter _masks --n_epochs 4000 --pretrained_model None --learning_rate 0.1 --diameter 0 --batch_size 16 --omni --RADAM`

On our bacterial phase contrast data, we found that while Cellpose does not benefit much from more than 500 epochs, but Omnipose continues to improve until around 4000 epochs. Omnipose outperforms Cellpose at 500 epochs but is significantly better at 4000. You can use `--save_every <n>` and `--save_each` to store intermediate model training states to explore this behavior. 


### 3D Omnipose

To train a 3D model on image volumes, used the dimension argument: `--dim 3`. You may have to choose a smaller crop size for images sent to the GPU. In that case, you can specify the size of the crop, *e.g.*, `--tyx 50,50,50`.


### Known limitations
Cell size remains the only practical limitation of Omnipose. On the low end, cells need to be at least 3 pixels wide in each dimension. On the high end, 60px appears to work well, with 150px being too large. The current workaround is to first downscale your images so that cells are within an appropriate size range (3-60px). This can be done automatically during training with `--diameter <X>`. The mean cell diameter `D` is calculated from the ground truth and images are rescaled by X/D. 

### Licensing
See `LICENSE.txt` for details. This license does not affect anyone using Omnipose for noncommercial applications. 
