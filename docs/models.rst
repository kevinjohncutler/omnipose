.. include:: sinebow.rst

:sinebow17:`Models`
==============================

All 2D models originally published in the Cellpose and Omnipose papers use :py:`nchan=2`. This is because Cellpose defaults are set to train models that use two channels for segmentation (usually cytoplasm and nucleus). Images without a second channel are just padded with :py:`0`\s. I think most users will train Omnipose on mono-channel images, so now :py:`nchan=1` by default.

.. tip:: 
    Always specify ``nchan`` and ``nclasses`` when training and evaluating models. 


Omnipose used to have a boundary prediction, so :py:`nclasses=3` (flow field, distance field, and boundary field in 2D). The current version of Omnipose no longer needs a boundary prediction, so :py:`nclasses=2` is the default. 

See the table below for named models and their corresponding ``nchan``, ``nclasses``. 

.. _pretrained-models:

:header-3:`Pretrained models`
-----------------------------
.. list-table:: 
    :header-rows: 1

    *   - model
        - ``nchan``
        - ``nclasses``
        - ``dim``
    *   - ``bact_phase_omni``
        - 2
        - 3
        - 2
    *   - ``bact_fluor_omni``
        - 2
        - 3
        - 2
    *   - ``cyto2_omni``
        - 2
        - 3
        - 2
    *   - ``worm_omni``
        - 2
        - 3
        - 2
    *   - ``plant_omni``
        - 2
        - 3
        - 3
    *   - ``bact_phase_omni_2``
        - 1
        - 2
        - 2


Cellpose models all have :py:`nchan=2`, :py:`nclasses=2`, and :py:`dim=2` (3D Cellpose uses 2D models to approximate 3D output). This means that if you wanted to, you could train an Omnipose model based on a Cellpose model using these hyperparameters (see :ref:`transfer-learning`). 




