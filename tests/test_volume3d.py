"""Headless tests for the 3D viewer payload engine (omnipose.gui._volume3d).

Runs against the spacetime test stack when available (crop for speed) plus
synthetic volumes. No model / server / browser / GPU required.
"""
import os

import numpy as np
import pytest

from omnipose.gui import _volume3d as v3

SPACE = "/Volumes/DataDrive/3D_spacetime/linked/a_baylii"
MASKS = os.path.join(SPACE, "dnaA_xy1_crop_masks.tif")
RAW = os.path.join(SPACE, "dnaA_xy1_crop.tif")
LINKS = os.path.join(SPACE, "dnaA_xy1_crop_links.txt")

have_stack = os.path.exists(MASKS)
needs_stack = pytest.mark.skipif(not have_stack, reason="spacetime stack not mounted")


def _synth_volume(D=6, H=24, W=24):
    """Two moving blobs across D frames -> a tiny labelled volume."""
    m = np.zeros((D, H, W), np.int32)
    for d in range(D):
        cy = 6 + d
        m[d, cy - 2:cy + 2, 5:9] = 1
        m[d, 14:18, 14 + d // 2:18 + d // 2] = 2
    return m


# --- kernel / steps --------------------------------------------------------

def test_kernel_steps_dims():
    s2, c2 = v3.kernel_steps(2)
    s3, c3 = v3.kernel_steps(3)
    assert s2.shape == (8, 2) and s2.dtype == np.int8
    assert s3.shape == (26, 3) and s3.dtype == np.int8
    # centre offset (all zeros) must be absent
    assert not (np.all(s3 == 0, axis=1)).any()
    # steps come in +/- pairs
    sset = {tuple(r) for r in s3.tolist()}
    assert all(tuple((-np.array(r)).tolist()) in sset for r in sset)


# --- encode / decode -------------------------------------------------------

@pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.uint32, np.float16, np.float32])
def test_encode_decode_roundtrip(dt):
    a = (np.random.default_rng(0).random((4, 5, 6)) * 50).astype(dt)
    for gz in (True, False):
        d = v3.encode_array(a, gzip_it=gz)
        b = v3.decode_array(d)
        assert b.shape == a.shape and b.dtype == a.dtype
        assert np.array_equal(a, b)


def test_label_dtype_narrowing():
    assert v3._narrow_label_dtype(np.array([[0, 40]])).dtype == np.uint8
    assert v3._narrow_label_dtype(np.array([[0, 300]])).dtype == np.uint16
    assert v3._narrow_label_dtype(np.array([[0, 70000]])).dtype == np.uint32


# --- flow / dist / affinity (synthetic) ------------------------------------

def test_flow_and_dist_shapes_synth():
    m = _synth_volume()
    mu, dist = v3.flow_and_dist(m, use_gpu=False)
    assert mu.shape == (3, *m.shape) and mu.dtype == np.float32
    assert dist.shape == m.shape
    assert np.isfinite(mu).all()


def test_affinity_volume_synth():
    m = _synth_volume()
    steps, spatial = v3.affinity_volume(m)
    assert steps.shape == (26, 3)
    assert spatial.shape == (26, *m.shape)
    assert spatial.dtype == np.uint8
    assert set(np.unique(spatial).tolist()) <= {0, 1}
    # affinity is 0 in background voxels
    bg = m == 0
    assert spatial[:, bg].sum() == 0


def test_flow_rgb_and_3d_shapes():
    m = _synth_volume()
    mu, dist = v3.flow_and_dist(m, use_gpu=False)
    rgb = v3.flow_rgb_slices(mu)
    rgb3 = v3.rgb_flow_3d(mu)
    drgb = v3.dist_rgb_slices(dist)
    assert rgb.shape == (*m.shape, 3) and rgb.dtype == np.uint8
    assert rgb3.shape == (*m.shape, 3) and rgb3.dtype == np.uint8
    assert drgb.shape == (*m.shape, 3) and drgb.dtype == np.uint8


# --- trajectories ----------------------------------------------------------

def test_trajectories_synth():
    m = _synth_volume(D=6)
    tr = v3.trajectories(m, edges=[[1, 2]])
    labels = {t["label"] for t in tr["tracks"]}
    assert labels == {1, 2}
    for t in tr["tracks"]:
        assert len(t["frames"]) == len(t["centroids"]) == 6
        for (y, x) in t["centroids"]:
            assert 0 <= y < m.shape[1] and 0 <= x < m.shape[2]
    assert tr["edges"] == [[1, 2]]
    # blob 1 moves down in y across frames -> centroid y increases
    t1 = next(t for t in tr["tracks"] if t["label"] == 1)
    ys = [c[0] for c in t1["centroids"]]
    assert ys[-1] > ys[0]


# --- spacetime stack (real data) -------------------------------------------

@needs_stack
def test_parse_links_real():
    edges = v3.parse_links(LINKS)
    assert len(edges) == 36
    assert edges[0] == [1, 7]
    # every parent splits into exactly 2 daughters (division)
    from collections import Counter
    counts = Counter(a for a, _ in edges)
    assert all(c == 2 for c in counts.values())


@needs_stack
def test_build_bundle_real_crop():
    import tifffile
    m = tifffile.imread(MASKS)[:8, 120:200, 120:200].astype(np.int32)
    vol = tifffile.imread(RAW)[:8, 120:200, 120:200]
    b = v3.build_bundle(vol, m, links_path=LINKS, use_gpu=False)

    assert b["meta"]["dim"] == 3
    assert b["meta"]["axes"] == ["t", "y", "x"]
    assert b["meta"]["depth"] == 8

    # mask roundtrips exactly
    mask_back = v3.decode_array(b["mask"])
    assert np.array_equal(mask_back.astype(np.int32), m)
    assert mask_back.dtype == np.uint8  # 40 labels -> uint8

    # image roundtrips
    img_back = v3.decode_array(b["image"])
    assert np.array_equal(img_back, vol)

    # flow payloads decode to right shapes
    raw = v3.decode_array(b["flow"]["raw"])
    assert raw.shape == (3, *m.shape)
    rgb = v3.decode_array(b["flow"]["rgbSlices"])
    assert rgb.shape == (*m.shape, 3)

    # affinity
    assert b["affinity"]["stepCount"] == 26
    sp = v3.decode_array(b["affinity"]["spatial"])
    assert sp.shape == (26, *m.shape)

    # trajectories: tracks for the founder labels present in the crop
    tr = b["trajectories"]
    present = set(np.unique(m).tolist()) - {0}
    track_labels = {t["label"] for t in tr["tracks"]}
    assert present <= track_labels
    assert len(tr["edges"]) == 36


@needs_stack
def test_bundle_from_files_autodetects_links():
    # full mask read is fine; flow/affinity off keeps it fast
    b = v3.bundle_from_files(None, MASKS, do_flow=False, do_affinity=False)
    assert b["meta"]["depth"] == 133
    assert len(b["trajectories"]["edges"]) == 36  # links auto-detected


def test_segmenter_build_volume_bundle_delegates():
    from omnipose.gui._segmenter import Segmenter
    seg = Segmenter()
    m = _synth_volume()
    b = seg.build_volume_bundle(None, m, do_flow=True, do_affinity=True,
                                do_trajectories=True)
    assert b["meta"]["dim"] == 3
    assert b["affinity"]["stepCount"] == 26
    assert v3.decode_array(b["flow"]["raw"]).shape == (3, *m.shape)

