.. include:: sinebow.rst


:sinebow11:`Installation`
=========================

.. include:: ../README.md
   :parser: myst_parser.sphinx_
   :start-after: ## How to install Omnipose
   :end-before: ### Python compatibility 

:header-2:`Python compatibility`
--------------------------------
.. include:: ../README.md
   :parser: myst_parser.sphinx_
   :start-after: ### Python compatibility 
   :end-before: ### Pyenv versus Conda

:header-2:`Pyenv versus Conda`
------------------------------
.. include:: ../README.md
   :parser: myst_parser.sphinx_
   :start-after: ### Pyenv versus Conda
   :end-before: ### GPU support 

:header-2:`GPU Support`
-----------------------
.. include:: ../README.md
   :parser: myst_parser.sphinx_
   :start-after: ### GPU support
   :end-before: ## How to use Omnipose


:header-2:`Where are models stored?`
------------------------------------
To maintain compatibility with Cellpose, the pretrained Omnipose models are also downloaded to ``$HOME/.cellpose/models/``.
This path on linux is ``/home/USERNAME/.cellpose/``, on macOS ``/Users/USERNAME/.cellpose/``, and on Windows
``C:\Users\USERNAME\.cellpose\models\``. These models are downloaded the first time you 
try to use them, either on the command line, in the GUI, or in a notebook.

If you would like to download the models to a different directory
and are using the command line or the GUI, 
you will need to always set the environment variable ``CELLPOSE_LOCAL_MODELS_PATH`` before you run ``python -m omnipose ...``
(thanks Chris Roat for implementing this!).

To set the environment variable in the command line/Anaconda prompt on windows run the following command modified for your path:
``set CELLPOSE_LOCAL_MODELS_PATH=C:/PATH_FOR_MODELS/``. To set the environment variable in the command line on 
linux, run ``export CELLPOSE_LOCAL_MODELS_PATH=/PATH_FOR_MODELS/``.

To set this environment variable when running Omnipose in a jupyter notebook, run 
this code at the beginning of your notebook before you import Omnipose:

::

   import os 
   os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = "/PATH_FOR_MODELS/"

:header-2:`Common issues`
-------------------------

If you receive the error: ``Illegal instruction (core dumped)``, then
likely mxnet does not recognize your MKL version. Please uninstall and
reinstall mxnet without mkl:

::

   pip uninstall mxnet-mkl
   pip uninstall mxnet
   pip install mxnet==1.4.0

If you receive the error: ``No module named PyQt5.sip``, then try
uninstalling and reinstalling pyqt5

::

   pip uninstall pyqt5 pyqt5-tools
   pip install pyqt5 pyqt5-tools pyqt5.sip

If you have errors related to OpenMP and libiomp5, then try 

::

   conda install nomkl

If you receive an error associated with **matplotlib**, try upgrading
it:

::

   pip install matplotlib --upgrade

If you receive the error: ``ImportError: _arpack DLL load failed``, then try uninstalling and reinstalling scipy
::

   pip uninstall scipy
   pip install scipy

If you are having issues with the graphical interface, make sure you have **python 3.8.5** installed. Higher versions *should* also work. 

If you are on macOS Yosemite or earlier, PyQt does not work and you won't be able
to use the GUI. More recent versions of macOS are fine. The software has been heavily tested on Windows 10 and
Ubuntu 18.04, and less well tested on macOS. Please post an issue if
you have installation problems.


.. :header-2:`Dependencies`
.. ------------------------

.. Omnipose relies on the following packages (which are
.. automatically installed with conda/pip if missing):

.. -  `pytorch`_
.. -  `pyqtgraph`_
.. -  `PyQt6`_
.. -  `numpy`_ (>=1.22.4)
.. -  `numba`_
.. -  `scipy`_
.. -  `scikit-image`_
.. -  `natsort`_
.. -  `matplotlib`_
.. -  sklearn_



.. .. _pyqtgraph: http://pyqtgraph.org/
.. .. _PyQt6: http://pyqt.sourceforge.net/Docs/PyQt6/
.. .. _numpy: http://www.numpy.org/
.. .. _numba: http://numba.pydata.org/numba-doc/latest/user/5minguide.html
.. .. _scipy: https://www.scipy.org/
.. .. _scikit-image: https://scikit-image.org/
.. .. _natsort: https://natsort.readthedocs.io/en/master/
.. .. _matplotlib: https://matplotlib.org/
.. .. _sklearn: https://scikit-learn.org/stable/




