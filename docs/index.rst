.. include:: sinebow.rst

|

.. image:: ../logo3.png
    :align: right
    :height: 300

.. include:: ../README.md
   :parser: myst_parser.sphinx_
   :start-after: io/py/omnipose)
   :end-before: ## Try out Omnipose online

Here we provide both the documentation for Omnipose 
and our fork of Cellpose. Please note this documentation is actively in development. 
For support, submit an `issue`_ on the Omnipose repo. For more on the workings of cellpose, check out our 
`twitter`_ thread and read the `paper`_.

| 

.. grid:: 1 1 2 2
    :gutter: 1 
    

    .. grid-item:: 

        .. grid:: 1 1 1 1
            :gutter: 0

            .. grid-item-card::

                .. toctree::
                    :maxdepth: 3
                    :caption: Basics
                    
                    installation
                    gui
                    inputs
                    settings
                    outputs
                    train

    
    .. grid-item::

        .. grid:: 1 1 1 1
            :gutter: 2

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

            .. grid-item-card::

                .. toctree::
                    :caption: nerd out
                    :maxdepth: 2

                    ncolor
                    diameter
                    gamma
