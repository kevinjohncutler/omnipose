.. include:: sinebow.rst

|

.. image:: ../logo3.png
    :class: no-lightbox
    :align: right
    :height: 300

.. include:: ../README.rst
   :start-after: |PyPI version|
   :end-before: Try out Omnipose online

Here we provide both the documentation for Omnipose 
and our fork of Cellpose. Please note this documentation is actively in development. 
For support, submit an `issue`_ on the Omnipose repo. For more on the workings of cellpose, check out our 
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
            gui
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
