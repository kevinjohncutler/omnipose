"""Centralized imports for the plot subpackage.

Generic plotting primitives (``figure``, ``imshow``, ``colorize``,
``normalize99``) live in ``ocdkit.plot`` and are re-exported through this
package's ``__init__.py``. This module just hosts the heavy stdlib /
third-party deps shared across leaves.
"""

import numpy as np
