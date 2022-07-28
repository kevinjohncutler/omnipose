.. include:: sinebow.rst

|

:sinebow20:`N-color`
=============================

If you have :math:`K` cells in an image and assign each a unique color, you will find for large :math:`K` (~10) that colors start to look too similar to distinguish 
from each other, even when using a colormap covering the entire visible spectrum (or in practice, the sRGB color space). Outline overlays are a good 
alternative for viewing segmentation results, but not for editing them (one must know to which particular cell each outline pixel belongs). Multiple similar 
colors can accidentally get used while editing the *wrong cell* (*e.g.*, color 11 inside cell 12) and ruin the segmentation despite this error being imperceptible 
to the human eye (this may account for many of the "errant pixels" we observe across ground-truth datasets of dense cells). 

To solve this problem, I developed the `ncolor`_ package, which converts K-integer label matrices to :math:`N \ll K` - color labels. The `four color theorem`_ 
guarantees that you only need 4 unique cell labels to cover all cells, but my algorithm opts to use 5 if a solution using 4 is not found quickly.
This was integral in developing the BPCIS dataset, and I subsequently incorporated it into Cellpose and Omnipose. By default, the GUI and plot commands display N-color 
masks for easier visualization. 

Interesting note: my code works for 3D volume labels as well, but there is no analogous theorem guaranteeing any sort of upper bound :math:`N<K` in 3D. 
In 3D, you could in principle have cells that touch every other cell, in which case :math:`N=K` and you cannot "recolor your map". On the dense but otherwise 
well-behaved volumes I have tested, my algorithm ends up needing 6-7 unique labels. I am curious if some bound on N can be formulated in the context of constrained volumes,
*e.g.*, packed spheres of mixed and arbitrary diameter...


test section

.. role:: raw-html(raw)
   :format: html

:raw-html:`<font color=red>test</font>`

:raw-html:`<font color=sinebow11>test</font>`

.. role:: sinebow11
:sinebow11:`test`