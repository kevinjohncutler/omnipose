.. include:: sinebow.rst

:sinebow19:`Command line`
=========================

Running just ``omnipose`` in the command line interface will launch the :doc:`gui`. I have left :doc:`training <training>` new models - done exclusively via CLI - to its own page. The rest of this page refers to evaluation on the command line. 

The command line allows batch processing and easy integration into
downstream analysis pipelines like SuperSegger, Morphometrics, MicrobeJ,
CellTool, and many others (any program that takes images and labels in
directories). See :doc:`settings` for an introduction to the
settings. The command line interface accepts parameters from
:mod:`cellpose_omni.models` for evaluation and from
:mod:`cellpose_omni.io` for finding files and saving output.

:header-2:`How to segment images using CLI`
-------------------------------------------

.. note:: 

   ``omnipose`` or ``python -m omnipose`` is equivalent to
   ``python -m cellpose --omni``, as our fork of Cellpose still provides
   the main framework for running Omnipose.

Run ``omnipose [arguments]`` and specify the arguments as follows. For
instance, to run on a folder with images where cytoplasm is green and
nucleus is blue and save the output as a png (using default diameter
30):

.. code-block:: 

    omnipose --dir <img_dir> --pretrained_model cyto –-chan 2 --chan2 3 --save_png


To do the same segmentation as in
:doc:`mono_channel_bact.ipynb <examples/mono_channel_bact>`, and
save TIF masks (this turns off cp_output PNGs) to folders along with
flows and outlines, run:

.. code-block:: 

    omnipose --dir <img_dir> –-use_gpu --pretrained_model bact_phase_omni \ 
             –-save_flows  –-save_outlines --save_tif –-in_folders 


Rescaling for the ``*bact*`` models is disabled by default, but setting
a diameter with the ``--diameter`` flag will rescale relative to 30px
(*e.g.* ``--diameter 15`` would double the image in x and y before
running the network).

.. warning::
   The path given to ``--dir`` must be an absolute path.

:header-2:`Recommendations`
---------------------------

There are some optional settings you should consider:

.. code-block:: 

    –-dir_above –-in_folders –-save_tifs –-save_flows –-save_outlines –-save_ncolor –-no_npy 


The ``--no_npy`` command just gets rid of the ``.npy`` output that many
users do not need. ``--save_tifs``, as an alternative to
``--save_pngs``, does not save the four-panel plot output (that can take
up a lot of space). Personally, I prefer to use ``--save_outlines`` when
I want a whole folder of easy-to-visualize segmentation results and
``--save_flows`` when I want to debug them. These are also nice to have
for making GIFs of cell growth, for example. ``--save_ncolor`` is handy
for exporting :doc:`ncolor` masks that are easier to edit by hand -
but it is the 1-channel version, no RGB colormap applied (which is what
you want for editing in Photoshop).

Most of all, ``--in_folders`` is something I always use so that these
various outputs do not clutter up the image directory (``/image01.png``,
``/image01_masks.tif``, ``/image01_flows.tif``\ …) and instead dumps all
the masks into a ``/masks`` folder, flows into ``flows``, N-color masks
into ``/ncolor``, outlines into ``/outlines``, and so on. Without the
``--dir_above`` command, these are inside the image directory.
``--dir_above`` will put those folders one directory above, parallel to
the image directory, which is what I like and what
`SuperSegger <https://github.com/tlo-bot/supersegger-omnipose>`__
expects.

``flow_threshold 0`` is a very good idea if you have a lot of large
images and do not need that cleanup step. Settings like
``--mask_threshold 0.3`` (0 is the default) can also be relevant. The
:doc:`gui` will automatically generate the parameters you need to
recapitulate your results in CLI (just in notebook formatting for now -
you will need to format those parameters according to these examples).

:header-2:`All options`
-----------------------

You can print out the full list of features with ``omnipose -h``. There are a lot
of them, but with Omnipose we organized them into categories. See
:doc:`CLI <cli>` to browse a bit easier. As demonstrated above,
``input image arguments`` and ``output arguments`` are the most
relevant. See
`SuperSegger-Omnipose <https://github.com/tlo-bot/supersegger-omnipose>`__
for an example of how to use these options to integrate Omnipose as a
segmentation backend.