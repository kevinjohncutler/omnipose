.. include:: sinebow.rst

:sinebow16:`Training`
=====================

Begin a training round in a terminal using the following command template:

.. code-block:: 

    omnipose --train --use_gpu --dir <training image directory> \
             --img_filter <img_filter> --mask_filter <mask_filter> \
             --nchan <nchan> --all_channels --channel_axis <channel_axis> \
             --pretrained_model None --diameter 0 --nclasses 2 \
             --learning_rate 0.1 --RAdam --batch_size 16 --n_epochs <n_epochs>
 

.. note:: 
    Training should be done only via CLI. If image preprocessing is required, I highly suggest doing that in a script and saving to a new folder (as opposed to attempting preprocessing + training in one script/notebook). 


The main commands here are:

:bash:`omnipose` 
    calls ``__main__.py`` in ``cellpose-omni``, which first loads the images in :bash:`--dir` and formats them. Then :bash:`--train` toggles on the training branch (versus evaluation). 
:bash:`--dir` 
    points to a folder of image and label pairs. With :bash:`--look_one_level_down`, you can let :bash:`--dir` point to a folder with subfolders. This can be very useful when training on several distinct subsets of ground truth data. 

:bash:`--diameter` 
    should be set to :py:`0` (and is now :py:`0` by default) to disable rescaling. Anything else will rescale your images relative to a mean diameter of 30 (see :doc:`diameter`), such that :bash:`--diameter 15` will **upscale** your image by a factor of 2 along each axis and :bash:`--diameter 60` will likewise **downscale** by a factor of 2. If you need automatic diameter estimation, see `Diameter and the Size Model`_. 

:bash:`--nchan`, :bash:`--nclasses` 
    define the number of image channels and the number of prediction classes. These should always be specified for **custom** models, as the defaults are `--nchan 1` (mono-channel images) and ``--nclasses 2`` (flow and distance field predictions). If you train a model with ``--nclasses 3`` (add the boundary field) or have multichannel images these will be in the model file name. Use these when running the model, too, both in CLI and in :mod:`cellpose_omni.models.CellposeModel()`. 

:bash:`--all_channels` 
    tells Omnipose to use all ``nchan`` channels for segmentation. The relatively complicated :bash:`--chan` and :bash:`--chan2` settings from Cellpose are still available, but I never use them. I highly recommend preprocessing your training set to have the channels you want to use (and for evaluation, do the same preprocessing in a script/notebook). 

:bash:`--channel_axis` 
    lets you specify where your channels are in your arrays. Conventional ordering is CYX for multichannel 2D images, 
    so :bash:`--channel_axis` defaults to :py:`0`. RGB images will have :bash:`--channel_axis 2`. 

.. warning:: 
    Paths given to :bash:`--dir` or :bash:`--test_dir` must absolute paths.


:header-2:`Hyperparameters`
---------------------------

It is best for reproducibility to explicitly choose hyperparameters at runtime rather than relying on defaults. 

:bash:`--RAdam` 
    selects the RAdam optimizer (versus the default SGD). I found RAdam to be a bit faster and more stable compared to SGD and other optimizers. 

:bash:`--learning_rate` 
    controls the optimizer step size. 

:bash:`--batch_size` 
    controls the number of images the network sees for each step (with the last batch being smaller if the number of images is not evenly divisible by :py:`batch_size`). A random crop is selected from each image (see :bash:`--tyx`). This means that only a portion of each image is seen during a given epoch. Smaller batches can sometimes lead to better generalization. Larger batches can lead to better stability. I have found that it does not make a very large difference in model performance, but larger batches can train faster (see :bash:`--dataparallel`). 

:bash:`--tyx` 
    controls the crop size for selecting a sample from each training image (see :ref:`image-dimensions`). 

:bash:`--n_epochs` 
    controls how many times the network is shown the full dataset. I usually do 4000. 

:bash:`--dataloader` 
    toggles on parallel dataloading. Preprocessing batches for training is a CPU bottleneck, but the DataParallel library helps a lot with that. Use :bash:`--num_workers` to control how many cores will participate. This is only a benefit when you have more images in your training set than cores on your machine. 

:header-2:`Model saving`
---------------------------
You can choose how often to save your models with ``--save_every <n>``. This overwrites the model every time. To save a new model each ``n`` epochs, you can use ``save_each`` (useful for debugging / comparing across epochs). 

    
:header-2:`Training data`
-------------------------
Your training set should consist of at least two tuples of images, labels, and (optionally) label link files. 

:header-3:`File naming conventions`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Each tuple of images and labels should be formatted as :py:`<base><img_filter>.<img_ext>`, :py:`<base><mask_filter>.<mask_ext>`, and (optionally) :py:`<base>_links.txt`. :py:`base` can be any string. The :py:`img_filter` defaults to an empty string :py:`''` and the  :py:`mask_filter` defaults to :py:`_masks`. These can be arranged in a single training folder: 

.. code-block:: text
    :class: h3

    folder/
    ├── A.tif
    ├── A_masks.tif
    ├── B.tif
    ├── B_masks.tif
    └── ...

Or in subfolders (when using :bash:`--look_one_level_down`):

.. code-block:: text
    :class: h3

    folder/
    ├── subfolder_1/
    │   ├── A.tif
    │   └── A_masks.tif
    ├── subfolder_2/
        ├── B.tif
        ├── B_masks.tif
        └── ...
    └── ...

If you use the :bash:`--img_filter` option (:bash:`--img_filter img` in this case), the suffix only goes on image files:

.. code-block:: text
    :class: h3

    folder/
    ├── A_img.tif
    ├── A_masks.tif
    ├── B_img.tif
    ├── B_masks.tif
    └── ...

:header-3:`File extensions`
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Microscopy images should generally be saved in a lossless format like PNG or TIF. Instance label matrices may likewise be stored as images in either PNG or TIF. Note that TIF supports up to 32 bits per channel whereas PNG only supports 16. That said, if you have more than :math:`2^{16}-1 = 65535` labels in one image, you should definitely be cropping your images into several smaller images. 

.. _image-dimensions:

:header-3:`Image dimensions`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You should aim to make training images of roughly size :py:`(512,512)`. During training, the :py:`tyx` parameter (set to :py:`224,224` by default) controls the size of warped image crops in each batch shown to the network. Although the true rectangular patch selected from each image in a batch has randomly expanded or contracted dimensions (within a range :py:`0.5-1.5`), you should aim to have the `tyx` dimensions roughly half that of the images in the training set. If much smaller, then each image will not be sufficiently covered during an epoch (requiring more epochs to converge). Larger ``tyx`` will just slow down training and possibly hurt generalizability. 

If an image dimension is substantially larger than 512 px, subdivide it along that axis. For example, :py:`(2048,2048)` images should be split into 16 :py:`(512,512)` images (4 along each axis). Smaller images are far easier to annotate correctly. 

If your image dimensions are substantially smaller than 512 px, you can instead decrease the :py:`tyx` parameter. For example, if your training images are around size :bash:`(256,256)`, then I would recommend the CLI flag :bash:`--tyx 128,128`. 

.. note:: 
    The `tyx` tuple elements must be evenly divisible by 8 (for U-net downsampling).  

:header-3:`Object density`
^^^^^^^^^^^^^^^^^^^^^^^^^^^
As a general rule, you want to train on images with densely packed objects. This is to balance the foreground class to the background class. In other words, we want Omnipose to focus on predicting good output in foreground regions rather than zero output in background regions. If your images have a lot of useless background, *crop out* just the denser regions. This can be done automatically if you can segment clusters/microcolonies of cells. You can use functions in :mod:`omnipose.utils` for processing a binary image into crops that you can then join into an ensemble image using a rectangle packing algorithm. Training on these images allows Omnipose to see the same number of cells but a lot faster, as it does not waste time looking at too much background. 


:header-3:`Ground truth quality`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
*Garbage in, garbage out.* It is better to have fewer images with meticulously crafted, consistent labels than many images with sloppy labels. Your labels should...

#. be based on supplemental channels wherever the primary channel is ambiguous
#. be label matrices, not semantic (binary) masks
#. not miss a single cell
#. extend to cell boundaries
#. meet each other at cell interfaces

You will probably spend 10x more time annotating ground truth images than acquiring them, so it is worth putting in the effort to find a membrane dye that does not conflict with main channel(s) on which your model will be trained. This is purely for the purposes of having a physiological reference for the ground truth of cell extent and cell septation, not for training the segmentation model. 

.. tip:: 
    If using a transmissive modality like phase contrast or brightfield or DIC, use the same filter cube as your fluorescence channel. This usually removes any offset between the channels. Otherwise, be sure to do multimodal registration between the channels. 

.. _transfer-learning:

:header-2:`Transfer learning`
-----------------------------

You can use :bash:`--pretrained_model None` to train from scratch or :bash:`--pretrained_model <model path>` to start from an existing model. Once a model is initialized and trained, you **cannot** change its structure. This is defined by :py:`nchan` (the number of channels used for segmentation), :py:`nclasses` (the number of prediction classes), and :py:`dim` (the dimension of the images). **You must use precisely the same** :py:`nchan`\ **,** :py:`nclasses`\ **, and** :py:`dim` **that were used to train the existing model.** See :doc:`models` for a table of the pretrained model parameters. 


.. _Diameter and the Size Model:
:header-2:`Diameter and the Size Model`
---------------------------------------

The Cellpose pretrained models are trained using resized images so that the cells have the same median diameter across all images.
If you choose to use a pretrained model, then this fixed median diameter is used. **Omnipose models are generally not trained with rescaling.** ``cyto2_omni`` is the exception, as its images are extremely diverse in size. 

If you choose to train from scratch, you can set the median diameter you want to use for rescaling with the :bash:`--diameter` flag, or set it to :py:`0` to disable rescaling. The ``cyto``, ``cyto2``, and ``cyto2_omni`` models were trained with a diameter of 30 pixels and the `nuclei` model was trained with a diameter of 17 pixels.

If your target image set varies a lot in cell diameter (i.e., the images you want to segment vary unpredictably in size), you may also want to learn a :mod:`~cellpose_omni.models.SizeModel()` that predicts the diameter from the network style vectors. Add the flag :bash:`--train_size` and this model will be trained and saved as an 
``*.npy`` file. **Omnipose models generally do not come with a** :mod:`~cellpose_omni.models.SizeModel()`, with the exception of ``cyto2_omni``.


:header-2:`Examples`
--------------------

To train on cytoplasmic images (green cyto and red nuclei) starting with a pretrained model from cellpose_omni (cyto or nuclei):

::
    
    omnipose --train --dir <train_path> --pretrained_model cyto --chan 2 --chan2 1

You can train from scratch as well:

::

    omnipose --train --dir <train_path> --pretrained_model None


You can also specify the full path to a pretrained model to use:

::

    omnipose --dir <train_path> --pretrained_model <model_path> --save_png


To train the ``bact_phase_omni`` model from scratch using the same parameters from the Omnipose paper, download the dataset and run

::

    omnipose --train --use_gpu --dir <bacterial_dataset_directory> --mask_filter _masks \ 
             --n_epochs 4000 --pretrained_model None --learning_rate 0.1 --diameter 0 \ 
             --batch_size 16  --RAdam --nclasses 3

:header-2:`Training 3D models`
------------------------------

.. include:: ../README.md
   :parser: myst_parser.sphinx_
   :start-after: 3D Omnipose
   :end-before: To evaluate Omnipose models on 3D data