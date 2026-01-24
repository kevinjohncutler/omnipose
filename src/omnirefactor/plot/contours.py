import numpy as np
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from skimage.segmentation import find_boundaries
from scipy.interpolate import splprep, splev

from ..utils import kernel_setup, get_neighbors
from ..core.affinity import boundary_to_masks, masks_to_affinity
from ..core.contour import get_contour


def vector_contours(fig, ax, mask, crop=None, smooth_factor=5, color="r", linewidth=1,
                    y_offset=0, x_offset=0,
                    pad=2,
                    mode="constant",
                    zorder=1,
                    ):

    msk = np.pad(mask, pad, mode="edge")

    if crop is not None:
        msk = msk[crop]

    msk = np.pad(msk, 1, mode="constant", constant_values=0)

    dim = msk.ndim
    shape = msk.shape
    steps, inds, idx, fact, sign = kernel_setup(dim)

    bd = find_boundaries(msk, mode="inner", connectivity=2)
    msk, bounds, _ = boundary_to_masks(bd, binary_mask=msk > 0, connectivity=1, min_size=0)

    coords = np.nonzero(msk)
    neighbors = get_neighbors(tuple(coords), steps, dim, shape)
    affinity_graph = masks_to_affinity(msk, coords, steps, inds, idx, fact, sign, dim, neighbors)

    contour_map, contour_list, unique_L = get_contour(
        msk, affinity_graph, coords, neighbors, cardinal_only=True
    )

    patches = []
    for contour in contour_list:
        if len(contour) > 1:
            pts = np.stack([c[contour] for c in coords]).T[:, ::-1]
            pts += np.array([x_offset, y_offset])
            tck, u = splprep(pts.T, u=None, s=len(pts) / smooth_factor, per=1)
            u_new = np.linspace(u.min(), u.max(), len(pts))
            x_new, y_new = splev(u_new, tck, der=0)

            if isinstance(pad, tuple):
                points = np.column_stack([x_new - (pad[0][0] + 1), y_new - (pad[1][0] + 1)])
            else:
                points = np.column_stack([x_new - (pad + 1), y_new - (pad + 1)])

            path = mpath.Path(points, closed=True)

            patch = mpatches.PathPatch(
                path,
                fill=None,
                edgecolor=color,
                linewidth=linewidth,
                zorder=zorder,
                capstyle="round",
            )
            patches.append(patch)

    if isinstance(ax, list):
        for a in ax:
            patch_collection = PatchCollection(patches, match_original=True, snap=False)
            a.add_collection(patch_collection)
    else:
        patch_collection = PatchCollection(patches, match_original=True, snap=False)
        ax.add_collection(patch_collection)
