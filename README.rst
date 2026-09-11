Omnipose
========

.. raw:: html

   <img src="https://github.com/kevinjohncutler/omnipose/blob/main/logo3.png?raw=true" width="200" title="bacteria" alt="bacteria" align="right" vspace = "0">
   <img src="https://github.com/kevinjohncutler/omnipose/blob/main/logo.png?raw=true" width="200" title="omnipose" alt="omnipose" align="center" vspace = "0">

|Downloads| |PyPI version|

.. _intro_start:

Omnipose is a general image segmentation tool that builds on
`Cellpose <https://github.com/MouseLand/cellpose>`__ in the ways
described in our
`paper <https://www.nature.com/articles/s41592-022-01639-4>`__. It works
for both 2D and 3D images and on any imaging modality or cell shape, so
long as you train it on representative images. We have several
pre-trained models for:

-  **bacterial phase contrast**: trained on a diverse range of bacterial
   species and morphologies.
-  **bacterial fluorescence**: trained on the subset of the phase data
   that had a membrane or cytosol tag.
-  **C. elegans**: trained on a couple OpenWorm videos and the
   `BBBC010 <https://bbbc.broadinstitute.org/BBBC010>`__ alive/dead
   assay.
-  **cyto2**: trained on user data submitted through the Cellpose GUI;
   useful as a fine-tuning starting point.

.. _intro_stop:

Use the GUI
-----------

Launch the desktop GUI from terminal: ``omnipose``. The 2.0 GUI is a
custom-built web viewer (FastAPI server + browser frontend, wrapped in
``pywebview`` for desktop) — not a fork of the Cellpose Qt GUI. The
segmentation panel exposes Omnipose-specific controls and the
`ncolor <https://github.com/kevinjohncutler/ncolor>`__ label representation
(default; toggle off to save standard mask labels).

The same server can run headless for remote use:

::

   omnipose --server --host 0.0.0.0 --port 8765

How to install Omnipose
-----------------------

.. _install_start:

The recommended setup is **pyenv + pip**. Installing pyenv itself
(homebrew on macOS, system build deps on Linux, ``pyenv-win`` on
Windows) is covered in the
`pyenv installation guide <https://github.com/pyenv/pyenv#installation>`__.
The short version:

1. Install pyenv for your platform.
2. Install a recent Python and set it as the global default:

   ::

      pyenv install 3.12.10
      pyenv global  3.12.10

3. Install the latest PyPi release of Omnipose:

   ::

      pip install omnipose

   or, for the development version (recommended during 2.0 release prep):

   ::

      git clone https://github.com/kevinjohncutler/omnipose.git
      cd omnipose
      pip install -e .

4. For the desktop GUI, add the ``[gui]`` extras:

   ::

      pip install -e .[gui]

This pulls ``imageio``, ``pywebview``, ``fastapi``, and ``uvicorn``.

Omnipose depends on
`ocdkit <https://github.com/kevinjohncutler/ocdkit>`__ for shared
utilities (array, GPU, I/O, measurement, morphology, spatial). It is
installed transparently as a dependency.

.. _install_stop:

.. warning::
   If you previously installed Omnipose, please run

   .. code-block::

      pip uninstall omnipose && pip cache remove omnipose

   to prevent version conflicts. See :ref:`project structure <project-structure>` for more details.


Python compatibility
~~~~~~~~~~~~~~~~~~~~

.. _python_start:

Omnipose 2.0 is tested on Python 3.10, 3.11, and 3.12; **3.12 is the
recommended target**. Earlier versions (3.8 / 3.9) may work but are no
longer exercised in CI. Use ``python -V`` to confirm which interpreter
your shell will pick up.

If you have multiple Python installs (e.g. a system Python alongside
pyenv), let pyenv shims sit at the front of ``$PATH`` so ``python`` and
``pip`` resolve to the pyenv version. ``which python`` should return
``~/.pyenv/shims/python``.

.. _python_stop:

Pyenv versus Conda
~~~~~~~~~~~~~~~~~~

.. _pyenv_start:

Pyenv is the recommended Python version manager for Omnipose. It is
faster to install than miniconda, faster at resolving environments,
plays well with napari (use ``pip install "napari[pyqt6]"`` to avoid Qt
conflicts), and handles Apple Silicon GPU (MPS) without the conda
gymnastics. Set your global version (3.10 – 3.12) and ``pip install
omnipose`` — pip pulls in the right PyTorch wheel for your platform.

Conda still works if you prefer it. Use
`miniforge <https://github.com/conda-forge/miniforge>`__ on Apple
Silicon. The README on GitHub has an expandable conda walkthrough.

.. _pyenv_stop:

GPU support
~~~~~~~~~~~

.. _gpu_start:

Omnipose runs on CPU on macOS, Windows, and Linux, and on GPU via:

- **NVIDIA CUDA** (Linux, Windows) — install the CUDA-enabled PyTorch
  wheel from the
  `official PyTorch selector <https://pytorch.org/get-started/locally/>`__
  *before* ``pip install omnipose``. Omnipose pins ``torch>=1.10`` only,
  so any compatible build is fine. Older drivers may be limited to older
  CUDA / PyTorch combinations — see
  `previous PyTorch versions <https://pytorch.org/get-started/previous-versions/>`__.
- **Apple Silicon MPS** (macOS arm64) — pip-installed PyTorch on Python
  3.10+ has full MPS support. End-to-end inference on MPS is roughly 2×
  faster in 2.0 than 1.x in our benchmarks. No extra steps.
- **AMD ROCm** (Linux) — should work via the ROCm PyTorch wheel, but is
  not routinely tested.

To confirm GPU availability:

::

   import torch
   torch.cuda.is_available()                    # NVIDIA
   torch.backends.mps.is_available()            # Apple Silicon

.. _gpu_stop:


.. _3d-omnipose:

3D Omnipose
-----------

To train a 3D model on image volumes, specify the dimension argument:
``--dim 3``. You may run out of VRAM on your GPU. In that case, you can
specify a smaller crop size, *e.g.*, ``--tyx 50,50,50``. The command used
in the paper on the *Arabidopsis thaliana* lateral root primordia
dataset was:

::

   omnipose --use_gpu --train --dir <path> --mask_filter _masks \
            --n_epochs 4000 --pretrained_model None --learning_rate 0.1 --save_every 50 \
            --save_each  --verbose --look_one_level_down --all_channels --dim 3 \
            --batch_size 4 --diameter 0 --nclasses 3

In 2.0, dynamic loss balancing is on by default, so the legacy
``--RAdam`` flag is no longer recommended (standard SGD converges faster
on bacterial phase data with the new loss).

.. _3d_omnipose_stop:

To evaluate Omnipose models on 3D data, see the
`examples <docs/examples/>`__. If you run out of GPU memory, consider
(a) evaluating on CPU or (b) using ``tile=True``.

Licensing
---------

See ``LICENSE`` for details. This license does not affect anyone
using Omnipose for noncommercial applications.

.. |Downloads| image:: https://static.pepy.tech/personalized-badge/omnipose?period=total&units=international_system&left_color=grey&right_color=green&left_text=Downloads
   :target: https://pepy.tech/project/omnipose
.. |PyPI version| image:: https://badge.fury.io/py/omnipose.svg
   :target: https://badge.fury.io/py/omnipose
