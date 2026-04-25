"""Test plot functions that require IPython/Jupyter via nbclient.

Coverage is collected inside the kernel and written to .coverage_combined/
so the cross-device script can combine it.
"""

import os
import pytest
import nbformat
from nbclient import NotebookClient

_COV_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '.coverage_combined'))
_SRC_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'src', 'omnipose'))


def _make_notebook(cells):
    os.makedirs(_COV_DIR, exist_ok=True)
    all_cells = [
        nbformat.v4.new_code_cell(
            "import coverage as _cov_mod; "
            f"_cov = _cov_mod.Coverage(source=[{_SRC_DIR!r}], "
            f"data_file={_COV_DIR!r} + '/.coverage.notebook', "
            "data_suffix=True); "
            "_cov.start()"
        ),
    ]
    all_cells.extend(nbformat.v4.new_code_cell(c) for c in cells)
    all_cells.append(nbformat.v4.new_code_cell("_cov.stop(); _cov.save()"))
    nb = nbformat.v4.new_notebook()
    nb.cells = all_cells
    return nb


def _run_notebook(nb, timeout=30):
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3")
    client.execute()
    return nb


class TestShowSegmentation:
    def test_basic(self):
        nb = _make_notebook([
            "import numpy as np",
            "from matplotlib.figure import Figure",
            "from omnipose.plot.display import show_segmentation",
            (
                "img = np.random.rand(32, 32).astype(np.float32)\n"
                "masks = np.zeros((32, 32), dtype=np.int32)\n"
                "masks[5:15, 5:15] = 1\n"
                "masks[18:28, 18:28] = 2\n"
                "flow = np.random.rand(32, 32, 3).astype(np.uint8)\n"
                "fig = Figure(figsize=(12, 3))\n"
                "fig.add_subplot(111)\n"
                "result = show_segmentation(fig, img, masks, flow, hold=True)\n"
                "assert result is not None"
            ),
        ])
        _run_notebook(nb)

    def test_multichannel(self):
        nb = _make_notebook([
            "import numpy as np",
            "from matplotlib.figure import Figure",
            "from omnipose.plot.display import show_segmentation",
            (
                "img = np.random.rand(32, 32, 2).astype(np.float32)\n"
                "masks = np.zeros((32, 32), dtype=np.int32)\n"
                "masks[5:15, 5:15] = 1\n"
                "flow = np.random.rand(32, 32, 3).astype(np.uint8)\n"
                "fig = Figure(figsize=(12, 3))\n"
                "fig.add_subplot(111)\n"
                "result = show_segmentation(fig, img, masks, flow, channels=[1, 2], hold=True)\n"
                "assert result is not None"
            ),
        ])
        _run_notebook(nb)
