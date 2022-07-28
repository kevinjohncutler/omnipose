.. include:: sinebow.rst

|

:sinebow13:`Inputs`
===================

Omnipose automatically detects TIFFs, PNGs, or JPEGs. ``cellpose.io`` uses tifffile or cv2 for image loading. 
Single-plane images can be formatted as nY x nX x channels or channels x nY x nX. 
The `channels <settings.html#channels>`__ settings will take care of reshaping 
the input appropriately for the network if we can safely assume that the smallest axis is the channel axis. 
For example, a 2 x 2048 x 2048 image will automatically have axis 0 set to be the channel axis. The `channel_axis` parameter
allows you to override this when necessary.

Note that Omnipose also rescales the input for 
each channel so that 0 = 0.01st percentile of image values and 1 = 99.99th percentile. 
These are not yet user-tunable parameters, but they will be in a future release. 



:header-2:`3D segmentation`
---------------------------

Multiple-plane and multiple-channel tiffs are supported in the GUI (can 
drag-and-drop) and are supported when running in a notebook.
Multiplane images should be of shape nplanes x channels x nY x nX or as 
nplanes x nY x nX. You can test this by running in python 

::

    import skimage.io
    data = skimage.io.imread('img.tif')
    print(data.shape)

If drag-and-drop of the tiff into 
the GUI does not work correctly, then it's likely that the shape of the tiff is 
incorrect. If drag-and-drop works (you can see a tiff with multiple planes), 
then the GUI will automatically run 3D segmentation and display it in the GUI. Watch 
the command line for progress. It is recommended to use a GPU to speed up processing.

If drag-and-drop doesn't work because of the shape of your tiff, 
you need to transpose the tiff and resave to use the GUI, or 
use the napari plugin for cellpose, or run CLI/notebook and 
specify the ``channel_axis`` and/or ``z_axis``
parameters:

    ``channel_axis`` and ``z_axis`` can be used to specify the axis (0-based) 
    of the image which corresponds to the image channels and to the z axis. 
    For example. a 105-plane z-stack image with 2 channels of shape (1024,1024,2,105,1) can be 
    specified with ``channel_axis=2`` and ``z_axis=3``. If ``channel_axis=None``, 
    cellpose will try to automatically determine the channel axis by choosing 
    the dimension with the minimal size after squeezing. If ``z_axis=None`` 
    cellpose will automatically select the first non-channel axis of the image 
    to be the Z axis (ZYX ordering). These parameters can be specified using the command line 
    with ``--channel_axis`` or ``--z_axis`` or as inputs to ``model.eval`` for 
    the ``Cellpose`` or ``CellposeModel`` model.

There are two distinct modes of 3D image processing. The first is Cellpose3D, which uses a 2D model on
orthogonal slices of the volume to estimate 3D predicitons from 2D network output. To use this in a notebook, 
set ``do_3D=True`. You can give a list of 3D inputs, or a single 3D/4D stack.
When running on the command line, add the flag ``--do_3D`` (it will run all tiffs 
in the folder as 3D tiffs if possible). 

If Cellpose3D segmentation is not working well and there is inhomogeneity in Z, try stitching 
masks in Z instead of running ``do_3D=True``. See details for this option here: 
`stitch_threshold <settings.html#d-settings>`__.

The second approach, implemented in Omnipose, is to directly predict 3D flows etc. by training 
models on 3D datasets. We offer one pretrained model: ``plant_omni``. The ``--dim`` argument allows users to
specify the dimensionality of their data/model for training and evaluation, so ``dim=2`` 
corresponds to 2D processing (even in Cellpose3D) and ``dim=3`` corresponds to 3D iprocessing. More work is needed to validate functionality of true 3D segmentation in the GUI. 



