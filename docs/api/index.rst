:hide-toc:

.. include:: ../sinebow.rst

:sinebow20:`API`
==========================

This page exists to help users navigate the labyrinth of functions and classes that make up Omnipose. 

.. _project-structure:

:header-2:`Project structure`
-----------------------------
.. include:: ../../README.rst
   :start-after: .. _ps1: 
   :end-before: .. _ps2:


.. Omnipose is composed of two main modules, ``core`` and ``utils``. The ``core`` module separates the truly new contributions of Omnipose to the Cellpose framework, while ``utils`` contains supporting functions that are either not needed in Cellpose or offer alternative / expanded functionality over similar functions within Cellpose. 

.. The API for our Cellpose fork is actively being expanded over that of the main branch, with the hope to eventually fill in all missing function descriptions. 

:header-2:`Modules`
--------------------------------
.. toctree::
    :maxdepth: 1
    :caption: Omnipose
    
    omnipose

.. toctree::
    :maxdepth: 1
    :caption: Cellpose
    
    cellpose

