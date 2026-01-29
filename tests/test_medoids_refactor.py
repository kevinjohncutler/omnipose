import numpy as np

from omnirefactor.metrics import scores
from omnirefactor.utils import medoids as medoids_mod


def _paint_circle(labels, center, radius, label_id):
    rs, ys, xs = scores._circle_mask(radius)
    mask = rs <= radius
    cy, cx = center
    yy = ys[mask] + cy
    xx = xs[mask] + cx
    valid = (
        (yy >= 0)
        & (yy < labels.shape[0])
        & (xx >= 0)
        & (xx < labels.shape[1])
    )
    labels[yy[valid], xx[valid]] = label_id


def test_get_medoids_circle_centers():
    labels = np.zeros((64, 64), dtype=np.int32)
    centers = [(16, 16), (16, 48), (48, 16), (48, 48)]
    radius = 5
    for i, center in enumerate(centers, start=1):
        _paint_circle(labels, center, radius, i)

    medoid_coords, medoid_labels = medoids_mod.get_medoids(labels, do_skel=False)
    assert medoid_coords is not None
    assert medoid_labels is not None

    by_label = {int(lbl): coord for lbl, coord in zip(medoid_labels, medoid_coords)}
    for label_id, center in enumerate(centers, start=1):
        coord = by_label[label_id]
        dist = np.linalg.norm(coord - np.array(center))
        assert dist <= 2.0
