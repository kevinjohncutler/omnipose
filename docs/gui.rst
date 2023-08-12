.. include:: sinebow.rst

:sinebow12:`GUI`
================
The Omnipose GUI is an expansion and refinement of that from Cellpose. It defaults to the ``bact_phase_omni`` model and corresponding model parameters. 
Additionally, we pre-load a small bacterial phase contrast image for demonstration purposes. Masks are also represented in :doc:`ncolor` format by default, which is handy for visualizing and editing. Be sure to untick the ``ncolor`` box to switch to standard label format before saving your masks if that format is 
what you need (what you see is what you get). 

.. note::
    The GUI only segments one image at a time, so it is really only intended for users to try out Omnipose and find the best model and optimal segmentation parameters with minimal setup. If you want to segment multiple images in a directory or train a model, use Omnipose in the :doc:`command line <command>` or a :doc:`jupyter notebook <notebook>`. The GUI prints out the current parameters for you in the bottom left. 


:header-2:`Starting the GUI`
----------------------------

The quickest way to start is to open the GUI from a command line terminal. You might need to open an anaconda prompt if you did not add anaconda to the path. Activate your omnipose conda environment and run ``omnipose`` (or ``python -m omnipose``). 

The first time Omnipose runs, it will ask you to download the GUI dependencies. When it finishes, run the launch command again. The terminal will remain open and you can see model download progress, error messages, etc.  as you interact with the GUI. 

You can **drag and drop** images (.tif, .png, .jpg, .gif) into the GUI and run Cellpose, and/or manually segment them. Omnipose waits to download a model until the first time you use it. When the GUI is processing, you will see the progress bar fill up and during this time you cannot click on anything in the GUI. For more information about what the GUI is doing you can look at the terminal/prompt with which you launched the GUI. For best accuracy and runtime performance, resize images so cells are less than 100 pixels across. 

For multi-channel, multi-Z tiffs, the expected format is ZCYX.

:header-2:`Using the GUI`
-------------------------

Main GUI mouse controls (works in all views):

-  Pan = left-click + drag
-  Zoom = scroll wheel (or +/= and - buttons)
-  Full view = double left-click
-  Select mask = left-click on mask
-  Delete mask = Ctrl (or Command on Mac) + left-click
-  Merge masks = Alt + left-click (will merge last two)
-  Start draw mask = right-click
-  End draw mask = right-click, or return to circle at beginning

Overlaps in masks are NOT allowed. If you draw a mask on top of another
mask, it is cropped so that it doesn't overlap with the old mask. Masks
in 2D should be single strokes (if *single_stroke* is checked).

If you want to draw masks in 3D, then you can turn *single_stroke*
option off and draw a stroke on each plane with the cell and then press
ENTER. 3D labeling will fill in unlabelled z-planes so that you do not
have to as densely label.

.. note::
    The GUI automatically saves after you draw a mask but NOT after
    segmentation and NOT after 3D mask drawing (too slow). Save in the file
    menu or with Ctrl+S. The output file is in the same folder as the loaded
    image with ``_seg.npy`` appended.

+---------------------+-----------------------------------------------+
| Keyboard shortcuts  | Description                                   |
+=====================+===============================================+
| CTRL+H              | help                                          |
+---------------------+-----------------------------------------------+            
| =/+  // -           | zoom in // zoom out                           |
+---------------------+-----------------------------------------------+
| CTRL+Z              | undo previously drawn mask/stroke             |
+---------------------+-----------------------------------------------+
| CTRL+0              | clear all masks                               |
+---------------------+-----------------------------------------------+
| CTRL+L              | load image (can alternatively drag and drop   |
|                     | image)                                        |
+---------------------+-----------------------------------------------+
| CTRL+S              | SAVE MASKS IN IMAGE to ``_seg.npy`` file      |
+---------------------+-----------------------------------------------+
| CTRL+P              | load ``_seg.npy`` file (note: it will load    |
|                     | automatically with image if it exists)        |
+---------------------+-----------------------------------------------+
| CTRL+M              | load masks file (must be same size as image   |
|                     | with 0 for NO mask, and 1,2,3... for masks)   |
+---------------------+-----------------------------------------------+
| CTRL+N              | load numpy stack (NOT WORKING ATM)            |
+---------------------+-----------------------------------------------+
| A/D or LEFT/RIGHT   | cycle through images in current directory     |
+---------------------+-----------------------------------------------+
| W/S or UP/DOWN      | change color (RGB/gray/red/green/blue)        |
+---------------------+-----------------------------------------------+
| PAGE-UP / PAGE-DOWN | change to flows and cell prob views (if       |
|                     | segmentation computed)                        |
+---------------------+-----------------------------------------------+
| , / .               | increase / decrease brush size for drawing    |
|                     | masks                                         |
+---------------------+-----------------------------------------------+
| X                   | turn masks ON or OFF                          |
+---------------------+-----------------------------------------------+
| Z                   | toggle outlines ON or OFF                     |
+---------------------+-----------------------------------------------+
| C                   | cycle through labels for image type (saved to |
|                     | ``_seg.npy``)                                 |
+---------------------+-----------------------------------------------+

:header-2:`Segmentation options`
--------------------------------

``SIZE``: you can manually enter the approximate diameter for your cells, or
press "calibrate" to let the SizeModel() estimate it. The size can be visualized
by a disk at the bottom of the view window (can turn this disk on by
checking "scale disk on"). Size defaults to 0 for bacterial models, which disables image resizing. 

``use GPU``: this will be grayed out for conda envoronemts / machines not configured for running pytorch on GPU. 

``MODEL``: choose among several pretrained models 

``CHAN TO SEG``: this is the channel in which the cytoplasm or nuclei exist

``CHAN2 (OPT)``: if *cyto** model is chosen, then choose the nuclear channel for this option




