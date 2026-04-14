.. include:: sinebow.rst

|

.. include:: ../README.rst
   :start-after: |PyPI version|
   :end-before: Try out Omnipose online

.. _project-structure:

Project structure
-----------------

See :doc:`api/index` for the module layout and entry points.

Here we provide documentation for Omnirefactor. Please note this documentation is actively in development.
For support, submit an `issue`_ on the Omnipose repo. For more on the workings of Omnipose, check out our
`twitter`_ thread and read the `paper`_.

| 

.. grid:: 1 1 2 2 
    :gutter: 2
    :margin: 2
    :padding: 0
    
    .. grid-item-card::

        .. toctree::
            :caption: Basics
            :maxdepth: 3
            
            installation
            inputs
            settings
            outputs
            training
            models


    .. grid-item::

        .. grid:: 1 1 1 1
            :gutter: 2
            :margin: 0 
            :padding: 0

            .. grid-item-card::

                .. toctree::
                    :caption: Examples
                    :maxdepth: 2

                    notebook
                    command

            .. grid-item-card::

                .. toctree::
                    :caption: Project
                    :maxdepth: 2
                    
                    api/index
                    cli

            .. grid-item-card::

                .. toctree::
                    :caption: More
                    :maxdepth: 2

                    affinity
                    contours
                    ncolor
                    diameter
                    gamma
                    logo

.. toctree::
    :hidden:

    timelapses
    train_via_notebook
