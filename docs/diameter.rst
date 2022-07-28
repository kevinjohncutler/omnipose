.. include:: sinebow.rst

|

:sinebow21:`Diameter metric`
============================

The idea of an average cell diameter sounds intuitive, but the standard implementation of this idea fails to capture that intuition. 
The go-to method (adopted in Cellpose) is to calculate the cell diameter as the diameter of the circle of equivalent area. As I will demonstrate, 
this fails for anisotropic (non-circular) cells. As an alternative, I devised the following simple diameter metric: 

``2*(n+1)*np.mean(dt_pos)``

Because the distance field represents the distance to the *closest* boundary point, it naturally captures the intrinsic 'thickness' of a region (in any dimension). Averaging the field over the region 
(the first moment of the distribution) distills this information into a number that is intuitively proportional to the thickness of the region. For example, if a region is made up of a bungle of many 
thin fragments, its mean distance is far smaller than the mean distance of the circle of equivalent area. But to call it a 'diameter', I wanted this metric to match the diameter of a sphere in any dimension. 
So, by calculating the average of distance field of an n-sphere, we get the above expression for the the diameter of an n-sphere given the average of the distance field over the volume. 

In the following example, bacterial cells exhibit constant width but increasing length. By plotting the mean diameter (averaged over all cells after being computed per-cell, of course), we find that 
the 'circle diameter metric' rises drastically with cell length, but my 'distance diameter metric' remains nearly constant. If we tried to use the former to train a ``SizeModel()``, images would get downsampled 
heavily to the point of cells being **too thin to segment**, and that is assuming that the model can reliably detect the highly nonlocal property of cell length in an image instead of the local property of 
cell width (at least, what we humans would point to and *call* cell width). 




