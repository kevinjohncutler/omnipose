from .imports import *

from ocdkit.measure import intersection_over_union


def stitch3D(masks, stitch_threshold=0.25):
    """Stitch 2D masks into 3D volume with stitch_threshold on IOU."""
    mmax = masks[0].max()
    empty = 0

    for i in range(len(masks) - 1):
        iou = intersection_over_union(masks[i + 1], masks[i])[1:, 1:]
        if not iou.size and empty == 0:
            masks[i + 1] = masks[i + 1]
            mmax = masks[i + 1].max()
        elif not iou.size and not empty == 0:
            icount = masks[i + 1].max()
            istitch = np.arange(mmax + 1, mmax + icount + 1, 1, int)
            mmax += icount
            istitch = np.append(np.array(0), istitch)
            masks[i + 1] = istitch[masks[i + 1]]
        else:
            iou[iou < stitch_threshold] = 0.0
            iou[iou < iou.max(axis=0)] = 0.0
            istitch = iou.argmax(axis=1) + 1
            ino = np.nonzero(iou.max(axis=1) == 0.0)[0]
            istitch[ino] = np.arange(mmax + 1, mmax + len(ino) + 1, 1, int)
            mmax += len(ino)
            istitch = np.append(np.array(0), istitch)
            masks[i + 1] = istitch[masks[i + 1]]
            empty = 1

    return masks