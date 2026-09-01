"""Centralized imports for the gui subpackage.

The omnipose GUI is an ``ocdkit.viewer`` plugin: ``ocdkit_plugin.py``
declares the plugin contract, ``_segmenter.py`` runs the actual
segmentation. Both share basic numpy + typing primitives.
"""

from typing import Any, Mapping

import numpy as np
