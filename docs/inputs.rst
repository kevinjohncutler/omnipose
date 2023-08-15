.. include:: sinebow.rst

:sinebow13:`Inputs`
===================

Omnipose automatically detects TIFs, PNGs, or JPEGs. Under the hood, :mod:`cellpose_omni.io` uses ``tifffile``
for loading TIFs and cv2 for PNG and JPEG. We are considering adding direct support for 
other bioformats types such as ND2, but for now all input must be exported to the above
image formats prior to running Omnipose. 


:header-2:`Channel formatting`
------------------------------
Single-plane, multichannel images can be formatted as :py:`(nY,nX,nChan)` or :py:`(nChan,nY,nX)`, 
the latter CYX formatting being more conventional and easier to work with (*e.g.*, in Napari). 
The :ref:`channels <channels>` settings will take care of reshaping 
the input appropriately for the network if we can safely assume that the smallest axis is the channel axis. 
For example, a :py:`(2,2048,2048)` image will automatically have axis :py:`0` set to be the channel axis. The `channel_axis` parameter
allows you to override this when necessary.

Note that Omnipose also rescales the input for 
each channel so that 0 = 0.01st percentile of image values and 1 = 99.99th percentile. 
These are not yet user-tunable parameters, but they will be in a future release. 


:header-2:`3D segmentation`
---------------------------

Multiple-plane and multiple-channel TIFs are supported in the GUI (can 
drag-and-drop) and are supported when running in a notebook.
Multiplane images should be of shape ZCYX or ZYX. You can test this by running in python:

::

    import skimage.io
    data = skimage.io.imread('img.tif')
    print(data.shape)

If drag-and-drop of the TIF into 
the GUI does not work correctly, then it's likely that the shape of the TIF is 
incorrect. If drag-and-drop works (you can see a TIF with multiple planes), 
then the GUI will automatically run 3D segmentation and display it in the GUI. Watch 
the command line for progress. It is recommended to use a GPU to speed up processing.

If drag-and-drop doesn't work because of the shape of your TIF, 
you need to transpose the TIF and re-save to use the GUI, or 
use the Napari plugin for Cellpose, or run CLI/notebook and 
specify the ``channel_axis`` and/or ``z_axis``
parameters:

    ``channel_axis`` and ``z_axis`` can be used to specify the axis (0-based) 
    of the image which corresponds to the image channels and to the z axis. 
    For example. a 105-plane z-stack image with 2 channels of shape :py:`(1024,1024,2,105,1)` can be 
    specified with :py:`channel_axis=2` and :py:`z_axis=3`. If :py:`channel_axis=None`, 
    cellpose will try to automatically determine the channel axis by choosing 
    the dimension with the minimal size after squeezing. If :py:`z_axis=None` 
    cellpose will automatically select the first non-channel axis of the image 
    to be the Z axis (ZYX ordering). These parameters can be specified using the command line 
    with :bash:`--channel_axis` or :bash:`--z_axis` or as inputs to ``model.eval`` for 
    the ``Cellpose`` or ``CellposeModel`` model.

There are two distinct modes of 3D image processing. The first is Cellpose3D, which uses a 2D model on
orthogonal slices of the volume to estimate 3D predicitons from 2D network output. To use this in a notebook, 
set :bash:`do_3D=True`. You can give a list of 3D inputs, or a single 3D/4D stack.
When running on the command line, add the flag :bash:`--do_3D` (it will run all TIFs 
in the folder as 3D TIFs if possible). 

If Cellpose3D segmentation is not working well and there is inhomogeneity in Z, try stitching 
masks in Z instead of running :py:`do_3D=True`. See details for this option here: 
`stitch_threshold <settings.html#d-settings>`__.

The second approach, implemented in Omnipose, is to directly predict 3D flows etc. by training 
models on 3D datasets. We offer one pretrained model: ``plant_omni``. The :bash:`--dim` argument allows users to
specify the dimensionality of their data/model for training and evaluation, so :py:`dim=2` corresponds to 2D 
processing (even in Cellpose3D) and :py:`dim=3` corresponds to 3D processing. 
More work is needed to validate functionality of true 3D segmentation in the GUI. 



