.. include:: sinebow.rst


:sinebow16:`Training`
=====================

.. note:: 
    Training is generally done only via CLI. We have reports of training not working via Jupyter notebook. 

Training for Omnipose works similar to Cellpose, but with extra augmentations and different hyperparameters, a different optimizer, and many new loss functions. 
All of these details are 'behind the scenes' - current Cellpose users just need to tweak their settings slightly to train an Omnipose model. 
Omnipose models **cannot** be trained starting from cellpose_omni models, so set ``--pretrained model None``. 

At the beginning of training, Cellpose computes the flow field representation for each 
mask image (:mod:`~cellpose.dynamics.labels_to_flows`). **Omnipose computes flows for training on-the-fly 
and no longer saves ground truth flows prior to training.**

The Cellpose pretrained models are trained using resized images so that the cells have the same median diameter across all images.
If you choose to use a pretrained model, then this fixed median diameter is used. **Omnipose models are generally not trained with rescaling** (``cyto2_omni`` is the exception). 

If you choose to train from scratch, you can set the median diameter you want to use for rescaling with the ``--diameter`` flag, or set it to 0 to disable rescaling. 
The ``cyto``, ``cyto2``, and ``cyto2_omni`` models were trained with a diameter of 30 pixels and the `nuclei` model with a diameter of 17 pixels.

If your training image set varies a lot in cell diameter, you may also want to learn a :mod:`~cellpose.models.SizeModel` that predicts the diameter from the styles that the 
network outputs. Add the flag ``--train_size`` and this model will be trained and saved as an 
``*.npy`` file. **Omnipose models generally do not come with a** :mod:`~cellpose.models.SizeModel`, with the exeption of ``cyto2_omni``.

The same channel settings apply for training models (see all Command line `options
<http://www.cellpose.org/static/docs/command.html>`_). 

Note that Cellpose expects the labelled masks (0=no mask, 1,2...=masks) in a separate file, e.g:

::

    wells_000.tif
    wells_000_masks.tif

If you use the --img_filter option (``--img_filter img`` in this case):

::

    wells_000_img.tif
    wells_000_masks.tif

.. warning:: 
    The path given to ``--dir`` and ``--test_dir`` must be an absolute path.

:header-2:`Training-specific options`
-------------------------------------

::

    --test_dir TEST_DIR       folder containing test data (optional)
    --n_epochs N_EPOCHS       number of epochs (default: 500)
  
To train on cytoplasmic images (green cyto and red nuclei) starting with a pretrained model from cellpose_omni (cyto or nuclei):

::
    
    python -m omnipose --train --dir ~/images_cyto/train/ --test_dir ~/images_cyto/test/ --pretrained_model cyto --chan 2 --chan2 1

You can train from scratch as well:

::

    python -m omnipose --train --dir ~/images_nuclei/train/ --pretrained_model None


You can also specify the full path to a pretrained model to use:

::

    python -m omnipose --dir ~/images_cyto/test/ --pretrained_model ~/images_cyto/test/model/cellpose_35_0 --save_png


To train the ``bact_phase_omni`` model from scratch using the same parameters from the Omnipose paper, download the dataset and run

::

    python -m omnipose --train --use_gpu --dir <bacterial dataset directory> --mask_filter _masks --n_epochs 4000 --pretrained_model None --learning_rate 0.1 --diameter 0 --batch_size 16  --RAdam

:header-2:`Training 3D models`
------------------------------

.. include:: ../README.md
   :parser: myst_parser.sphinx_
   :start-after: 3D Omnipose
   :end-before: To evaluate Omnipose models on 3D data