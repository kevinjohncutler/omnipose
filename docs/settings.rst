.. include:: sinebow.rst

:sinebow14:`Settings`
=====================

The most important settings are described on this page. 
See :mod:`cellpose_omni.models()` for all options. 

This is a typical example of using an Omnipose model to segment a list of images in a notebook. 
Cellpose users need only select an Omnipose model and use ``omni=True`` to update their existing code. 

::

    from cellpose_omni import models
    import skimage.io
    model = models.Cellpose(gpu=False, 
                            model_type='bact_phase_omni', 
                            nclasses=4, 
                            nchan=2, 
                            dim=2)

    files = ['img0.tif', 'img1.tif']
    imgs = [skimage.io.imread(f) for f in files]
    masks, flows, styles, diams = model.eval(imgs, 
                                             diameter=None, 
                                             channels=[0,0], 
                                             threshold=0.4, 
                                             omni=True)

This example shows the same settings used for each image, but you can also pass in a list for ``channels`` and ``diameter`` that specifies unique values to apply to each image. 
See our :doc:`example notebooks <notebook>`
for a solid introduction and figure notebooks for more advanced examples. 

.. tip::
    Use :py:`pretrained_model=<path to model>` in place of :py:`model_type=<model name>` when you want to use a model that is not built-in. 
    Specify :py:`nclasses` and :py:`nchan` if you encounter any issues in the model initialization (see :ref:`pretrained-models`). 

.. _channels:
:header-2:`Channels`
--------------------

Use ``channels = [0,0]`` for mono-channel images or multi-channel images that you would like converted to grayscale prior to segmentation. 
``[0,0]`` is what we used to train and evaluate our ``bact_phase_omni``, ``bact_fluor_omni``, ``worm_omni``, ``worm_high_res_omni``, and ``plant_omni`` models. 
If you do want to run segmentation on a specific channel of multi-channel images, use `1-based-indexing` ``[i,0]`` with ``i = 1,2,3,...`` for red, green, blue, ..., respectively. 
For example, you might have blue nuclei that look a lot like fluorescent bacteria, so could use the ``bact_fluor_omni`` model with ``channels = [2,0]``. 

You can also use two channels for segmentation: a cytoplasm channel and a nuclear channel. 
The ``cyto2_omni`` model was trained with image channels re-ordered to have red cytoplasm and green nucleus 
(where applicable in the dataset) using ``--chan 1 --chan2 2`` and therefore was evaluated using ``channels = [1,2]``.

See :doc:`mono_channel_bact.ipynb <../examples/mono_channel_bact>` for a monochannel segmentation on bacterial phase contrast images 
and :doc:`multi_channel_cyto.ipynb <../examples/multi_channel_cyto>` for multichannel segmentation of mouse neuron cells. 


:header-2:`Flow threshold`
--------------------------

The neural network may predict hallucinate network outputs that do not correspond well to the masks found by 
the mask reconstruction dynamics. As a consistency check, we can compute the 'true' flow field from the predicted labels and
compare this to the network predictions pixel-by-pixel. The ``flow_threshold`` parameter is the maximum allowed error of the flows 
averaged over all pixels in a given mask. The default is ``flow_threshold=0.4``. Increase this threshold 
if Omnipose is not returning as many masks as you expect. Decrease this threshold if Omnipose is returning too many 
spurious masks.

.. note::
    Well-trained models really don't need this and we set ``flow_threshold=0.0`` for most of our model evaluation. This disables the flow error calculation and will make Omnipose run a lot faster on large datasets. 

:header-2:`Mask threshold`
---------------------------

This threshold is applied to the distance transform output of Omnipose (or the cellprob output of Cellpose) 
to seed cell masks pixels for running dynamics. The default 
is ``mask_threshold=0.0``. Decrease this threshold if you are getting too few masks or if masks do not cover the entire cell. 

.. tip::
    The GUI provides sliders that update the Omnipose output for ``flow_threshold`` and ``mask_threshold`` in real time, which is very fast even on CPU for small images (~500 x 500 px). 


:header-2:`Diameter` 
--------------------

In most Omnipose models, we set ``diameter=0`` to disable image rescaling. We found that rescaling to a common cell diameter is only necessary when the images for training and evaluation 
have extreme diffrences in cell size, such as in the cyto2 dataset. Therefore, ``cyto2_omni`` was trained with a mean diameter of 30px just like the Cellpose ``cyto`` model. This means that 
images are rescaled by a factor of ``30.0/D`` where D is the mean diameter of all cells in the image. See the page on `mean cell diameter <diameter.html#diameter>`__ to see how Omnipose handles this better than Cellpose. 


The ``worm_high_res_omni`` is another example where rescaling was necessary. We suspect that it is the network architecture kernel size and number of down-sampling stages that prevents 
accurate prediction of boundary-derived output like flow and distance at the centers of objects. For these high-resolution *C. elegans* images, we found 60px to work well, but we did not 
do more tests to push this higher. To use this model, images should be rescaled by a factor of ``60.0/D``. 

.. tip::
    At this time, the diameter used for training is not saved with the model parameters and therefore must be specified using ``mymodel.diameter=60.0`` after initializing ``mymodel=models.CellposeModel()``. 
    30 is the default for models with `cyto` in the name but can be overwritten as shown. Similarly, `nuclei`-named models default to a mean diameter of 17 and `bacteria`-named models default to a mean diameter of 0 (rescaling disabled). 

:header-2:`SizeModel()`
---------------------------

In contrast to the ``CellposeModel()`` class that takes ``diameter`` as an option for rescaling, the ``Cellpose`` class includes a ``SizeModel()`` for automatic diameter estimation. 
This is a linear regression model trained on the 'style' vector of the network, which you can think of as a 64-dimensional summary of the input image. A SizeModel() for Omnipose was 
trained on the ``cyto2`` dataset to predict our own `cell diameter <diameter.html#diameter>`__ from the style vector. To use the ``SizeModel()``, we follow a two-step process:

1. Run the image through the cellpose network and obtain the style vector. Predict the size using the linear regression model from the style vector.
2. Resize the image based on the predicted size and run cellpose again, and produce masks. Take the final estimated size as the median diameter of the predicted masks.

For automated estimation in the ``Cellpose()`` class set ``diameter = None`` (default).
However, if this estimate is incorrect, you will need to set the diameter manually.

Changing the diameter will change the results that the algorithm 
outputs. When the diameter is set smaller than the true size 
then Omnipose may over-segment cells. Similarly, if the diameter 
is set too big then Omnipose may under-segment cells.

:header-2:`Resample`
--------------------

The cellpose network is run on your rescaled image -- where the rescaling factor is determined 
by the diameter you input (or determined automatically as above). For instance, if you have 
an image with 60 pixel diameter cells, the rescaling factor is 30./60. = 0.5. After network predictions are made,
the model runs the dynamics. The dynamics can be run at the rescaled 
size (``resample=False``), or the dynamics can be run on the resampled, interpolated flows 
at the true image size (``resample=True``). ``resample=True`` will create smoother masks when the 
cells are large but will be slower. ``resample=False`` can produce some jagged mask edges due to nearest-neighbor interpolation.
The default our Cellpose fork is ``resample==True``. 


:header-2:`3D settings`
-----------------------

Volumetric stacks do not always have the same sampling in XY as they do in Z. 
Therefore you can set an ``anisotropy`` parameter to allow for differences in 
sampling, e.g. set to 2.0 if Z is sampled half as dense as X or Y. 

There may be additional differences in YZ and XZ slices 
that make them unable to be used for 3D segmentation. 
I'd recommend viewing the volume in those dimensions if 
the segmentation is failing. In those instances, you may want to turn off 
3D segmentation (``do_3D=False``) and run instead with ``stitch_threshold>0``. 
Cellpose will create masks in 2D on each XY slice and then stitch them across 
slices if the IoU between the mask on the current slice and the next slice is 
greater than or equal to the ``stitch_threshold``. 

3D segmentation ignores the ``flow_threshold`` because we did not find that
it helped to filter out false positives in our test 3D cell volume. Instead, 
we found that setting ``min_size`` is a good way to remove false positives.





