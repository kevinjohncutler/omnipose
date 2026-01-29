import numpy as np

from omnirefactor.metrics import scores


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


def test_scores_shifted_masks():
    labels = np.zeros((64, 64), dtype=np.int32)
    centers = [(20, 20), (20, 44)]
    radius = 6
    for i, center in enumerate(centers, start=1):
        _paint_circle(labels, center, radius, i)

    # shift masks down by a few pixels to reduce IoU
    shifted = np.zeros_like(labels)
    shift = 3
    shifted[shift:] = labels[:-shift]

    ap, tp, fp, fn = scores.average_precision(labels, shifted, threshold=[0.9])
    assert ap.shape == (1,)
    assert 0.0 <= ap[0] < 1.0
    assert tp.shape == ap.shape
    assert fp.shape == ap.shape
    assert fn.shape == ap.shape

    aji = scores.aggregated_jaccard_index([labels], [shifted])
    assert aji.shape == (1,)
    assert 0.0 <= aji[0] < 1.0

    precision, recall, fscore = scores.boundary_scores([labels], [shifted], scales=[0.5, 1.0])
    assert precision.shape == (2, 1)
    assert recall.shape == (2, 1)
    assert fscore.shape == (2, 1)
    assert np.all(precision <= 1.0)
    assert np.all(recall <= 1.0)
    assert np.all(fscore <= 1.0)
