import numpy as np

from omnirefactor.measure import box as mbox
from omnirefactor.measure import crop as mcrop
from omnirefactor.measure import dist as mdist
from omnirefactor.measure import skeleton as mskeleton


def test_bartlett_nd_normalized():
    kernel_1d = mbox.bartlett_nd(5)
    assert kernel_1d.shape == (5,)
    assert np.isclose(kernel_1d.sum(), 1.0)

    kernel_2d = mbox.bartlett_nd((3, 5))
    assert kernel_2d.shape == (3, 5)
    assert np.isclose(kernel_2d.sum(), 1.0)


def test_find_highest_density_box_full_and_cluster():
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[2:5, 2:5] = 1
    labels[7:8, 7:8] = 1

    full = mbox.find_highest_density_box(labels, -1)
    assert full == (slice(0, 10), slice(0, 10))

    slc = mbox.find_highest_density_box(labels, 3)
    assert len(slc) == 2
    assert (slc[0].stop - slc[0].start) == 3
    assert (slc[1].stop - slc[1].start) == 3
    # box should include the densest cluster near the top-left
    assert slc[0].start <= 3 <= slc[0].stop
    assert slc[1].start <= 3 <= slc[1].stop


def test_bbox_to_slice_and_make_square():
    bbox = (2, 3, 6, 7)
    shape = (10, 10)
    slc = mcrop.bbox_to_slice(bbox, shape, pad=1, im_pad=1)
    assert slc[0].start >= 1
    assert slc[1].start >= 1
    assert slc[0].stop <= 9
    assert slc[1].stop <= 9

    square = mcrop.make_square(bbox, shape)
    assert (square[2] - square[0]) == (square[3] - square[1])
    assert 0 <= square[0] <= square[2] <= shape[0]
    assert 0 <= square[1] <= square[3] <= shape[1]


def test_crop_bbox_variants():
    mask = np.zeros((20, 20), dtype=np.int32)
    mask[2:6, 2:6] = 1
    mask[12:18, 12:18] = 2

    slices = mcrop.crop_bbox(mask, pad=1, iterations=1, area_cutoff=0, binary=False)
    assert len(slices) == 2

    biggest = mcrop.crop_bbox(mask, pad=1, iterations=1, get_biggest=True)
    assert len(biggest) == 1

    merged = mcrop.crop_bbox(mask, pad=1, iterations=1, binary=True)
    assert isinstance(merged, tuple)
    assert merged[0].start <= 2 and merged[0].stop >= 18
    assert merged[1].start <= 2 and merged[1].stop >= 18

    square = mcrop.crop_bbox(mask, pad=1, iterations=1, binary=True, square=True)
    assert (square[0].stop - square[0].start) == (square[1].stop - square[1].start)


def test_extract_patches_edges_and_point_order():
    img = np.arange(25, dtype=np.int32).reshape(5, 5)
    points = [(0, 0), (4, 4)]
    patches, slices = mcrop.extract_patches(img, points, box_size=3, fill_value=-1, point_order="yx")
    assert patches.shape == (2, 3, 3)
    assert patches[0, 0, 0] == -1
    assert patches[1, -1, -1] == -1
    assert len(slices) == 2

    patches_xy, _ = mcrop.extract_patches(img, [(0, 0)], box_size=3, fill_value=-1, point_order="xy")
    assert patches_xy.shape == (1, 3, 3)

    try:
        mcrop.extract_patches(img, [(0, 0)], box_size=3, point_order="bad")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for invalid point_order")


def test_distance_to_boundary_2d_and_3d():
    masks = np.zeros((10, 10), dtype=np.int32)
    masks[3:7, 3:7] = 1
    dist = mdist.distance_to_boundary(masks)
    assert dist.shape == masks.shape
    assert dist[masks > 0].max() > 0

    masks_3d = np.stack([masks, masks])
    dist_3d = mdist.distance_to_boundary(masks_3d)
    assert dist_3d.shape == masks_3d.shape

    try:
        mdist.distance_to_boundary(np.zeros((1,), dtype=np.int32))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for invalid ndim")


def test_find_boundaries_and_skeletonize():
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[2:6, 2:6] = 1
    labels[5:9, 5:9] = 2

    bd = mskeleton.find_boundaries(labels, connectivity=1, use_symmetry=False)
    bd_sym = mskeleton.find_boundaries(labels, connectivity=1, use_symmetry=True)
    assert bd.shape == labels.shape
    assert bd_sym.shape == labels.shape
    assert bd.sum() > 0
    assert bd_sym.sum() > 0

    skel = mskeleton.label_skeletonize(labels)
    assert skel.shape == labels.shape
    assert np.any(skel == 1)
    assert np.any(skel == 2)


def test_skeletonize_with_dt_and_extract():
    labels = np.zeros((7, 7), dtype=np.int32)
    labels[2:5, 2:5] = 1
    dt = np.zeros_like(labels, dtype=np.float32)
    dt[3, 3] = 2.0
    skel = mskeleton.skeletonize(labels, dt_thresh=1.0, dt=dt)
    assert skel[3, 3] == 1

    distance_field = np.zeros((5, 5), dtype=np.float32)
    distance_field[2, 2] = 1.0
    extracted = mskeleton.extract_skeleton(distance_field)
    assert extracted.shape[-2:] == distance_field.shape
    assert extracted.dtype == bool
