"""omnirefactor.plot — segmentation result visualization (Layer 3).

Layer 3: depends on L1 (transforms) and L2 (io).

Segmentation-specific code lives in leaf modules (``display``, ``edges``,
``overlay``). Generic plotting primitives live in ``ocdkit.plot`` and are
re-exported here so ``omnirefactor.plot.X`` keeps working for notebooks.
"""

import warnings

from ..load import enable_submodules
from ocdkit.plot import *  # canonical gateway for plot functions
from ..transforms.imports import normalize99, rescale
from ..io import masks_to_outlines


def dx_to_circ(dP, transparency=False, **kwargs):
    """Deprecated: use :func:`rgb_flow` instead."""
    warnings.warn(
        "dx_to_circ is deprecated, use rgb_flow instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return rgb_flow(dP, transparency=transparency)


apply_mpl_defaults()
enable_submodules(__name__)
