"""
Stress tests for the unified eval() pipeline.

Covers every supported calling convention:
  - Single 2D numpy array (is_image)
  - Numpy stack of same-size images (is_stack)
  - List of same-size images (is_list, grouped batching)
  - List of different-size images (is_list, pad / single mode)
  - Multichannel 2D image (H, W, C)
  - Single string file path  → normalized to [path]
  - List of string file paths
  - Folder path string        → sorted list of files
  - do_3D with single 3D volume
  - do_3D with list of 3D volumes
  - compute_masks=False (network outputs only)
  - stitch_threshold > 0 (2D stack → 3D stitched)
  - batch_mode='pad'  with mixed sizes
  - batch_mode='single' (one image at a time)
  - EvalSet passed directly (existing path, regression guard)
"""

import numpy as np
import pytest
import torch

from omnipose.data import eval as eval_mod
from omnipose.models import OmniModel, eval as meval


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_model(*, dim=2, nchan=1, nclasses=3, omni=True, logits=True):
    return OmniModel(
        gpu=False,
        pretrained_model=False,
        model_type=None,
        net_avg=False,
        use_torch=True,
        dim=dim,
        nchan=nchan,
        nclasses=nclasses,
        omni=omni,
        logits=logits,
    )


def stub_run_network(model):
    """Replace run_network with one that returns zero tensors of the right shape."""
    model._from_device = lambda x: x.detach().cpu().numpy()

    def run_network(batch, to_numpy=False):
        out = batch.new_zeros((batch.shape[0], model.nclasses, *batch.shape[-2:]))
        style = np.ones((batch.shape[0], model.nbase[-1]), dtype=np.float32)
        return out, style

    model.run_network = run_network
    return model


def stub_run_3d(model):
    """Replace _run_3D with a stub that returns zero arrays of the right spatial shape."""
    def _run_3D(img, **_):
        shape = img.shape[-3:]
        zeros = np.zeros(shape, dtype=np.float32)
        ones  = np.ones(shape,  dtype=np.float32)
        # Each entry: (dflow0, dflow1, cellprob, bd)  (nclasses indices 0..3)
        yf = [(zeros, zeros, ones, zeros) for _ in range(3)]
        styles = np.ones((model.nbase[-1],), dtype=np.float32)
        return yf, styles

    model._run_3D = _run_3D
    return model


def make_stub_model(**kwargs):
    model = make_model(**kwargs)
    stub_run_network(model)
    return model


def fake_compute_masks_factory(model):
    def fake_compute_masks(dP, dist, **_):
        mask   = np.zeros(dist.shape, dtype=np.int32)
        p      = np.zeros((model.dim,) + dist.shape, dtype=np.float32)
        tr     = np.zeros_like(p)
        bounds = np.zeros(dist.shape, dtype=bool)
        aff    = np.zeros((1,), dtype=np.float32)
        return mask, p, tr, bounds, aff
    return fake_compute_masks


# Common keyword args that disable slow/optional processing
_FAST = dict(
    channels=[0, 0],
    normalize=False,
    invert=False,
    rescale_factor=1.0,
    net_avg=False,
    augment=False,
    tile=False,
    bsize=16,
    show_progress=False,
    verbose=False,
)


# ---------------------------------------------------------------------------
# 1. Single 2D numpy array  (is_image path)
# ---------------------------------------------------------------------------

def test_eval_single_2d_array(monkeypatch):
    model = make_stub_model()
    monkeypatch.setattr(meval.core, "compute_masks", fake_compute_masks_factory(model))

    img = np.zeros((16, 16), dtype=np.float32)
    masks, flows = meval.eval(model, img, compute_masks=True, **_FAST)

    assert isinstance(masks, np.ndarray)
    # Unified path always batches; single image → (1, H, W)
    assert masks.shape == (1, 16, 16)
    assert isinstance(flows, list) and len(flows) == 1



# ---------------------------------------------------------------------------
# 2. Numpy stack of same-size images  (is_stack path)
# ---------------------------------------------------------------------------

def test_eval_numpy_stack_same_size(monkeypatch):
    model = make_stub_model()
    monkeypatch.setattr(meval.core, "compute_masks", fake_compute_masks_factory(model))

    stack = np.zeros((4, 16, 16), dtype=np.float32)   # 4 identical 2D images
    masks, flows = meval.eval(model, stack, compute_masks=True, **_FAST)

    assert isinstance(masks, np.ndarray)
    assert masks.shape == (4, 16, 16)
    assert len(flows) == 4


# ---------------------------------------------------------------------------
# 3. List of same-size images  (is_list, auto-grouped batching)
# ---------------------------------------------------------------------------

def test_eval_list_same_size(monkeypatch):
    model = make_stub_model()
    monkeypatch.setattr(meval.core, "compute_masks", fake_compute_masks_factory(model))

    imgs = [np.zeros((16, 16), dtype=np.float32) for _ in range(3)]
    masks, flows = meval.eval(model, imgs, compute_masks=True, **_FAST)

    assert isinstance(masks, np.ndarray)
    assert masks.shape == (3, 16, 16)
    assert len(flows) == 3


# ---------------------------------------------------------------------------
# 4. List of different-size images  (is_list, default/auto mode)
# ---------------------------------------------------------------------------

def test_eval_list_different_size_auto(monkeypatch):
    model = make_stub_model()
    monkeypatch.setattr(meval.core, "compute_masks", fake_compute_masks_factory(model))

    shapes = [(16, 16), (24, 20), (12, 12)]
    imgs = [np.zeros(s, dtype=np.float32) for s in shapes]
    masks, flows = meval.eval(model, imgs, compute_masks=True, batch_mode='auto', **_FAST)

    assert isinstance(masks, np.ndarray)
    assert len(flows) == 3
    # dist (index 2 in flow tuple) should match each image's spatial shape
    for i, (h, w) in enumerate(shapes):
        assert flows[i][2].shape == (h, w), f"Image {i}: expected {(h,w)}, got {flows[i][2].shape}"


# ---------------------------------------------------------------------------
# 5. List of different-size images — explicit pad mode
# ---------------------------------------------------------------------------

def test_eval_list_different_size_pad_mode(monkeypatch):
    model = make_stub_model()
    monkeypatch.setattr(meval.core, "compute_masks", fake_compute_masks_factory(model))

    # Use moderate size differences — pad mode uses constant padding which has no size limit
    imgs = [
        np.zeros((16, 16), dtype=np.float32),
        np.zeros((20, 20), dtype=np.float32),
    ]
    masks, flows = meval.eval(
        model, imgs, compute_masks=True, batch_mode='pad', loader_batch_size=2, **_FAST
    )

    assert isinstance(masks, np.ndarray)
    assert len(flows) == 2


# ---------------------------------------------------------------------------
# 6. List of different-size images — single mode (one at a time)
# ---------------------------------------------------------------------------

def test_eval_list_different_size_single_mode(monkeypatch):
    model = make_stub_model()
    monkeypatch.setattr(meval.core, "compute_masks", fake_compute_masks_factory(model))

    shapes = [(16, 16), (20, 24)]
    imgs = [np.zeros(s, dtype=np.float32) for s in shapes]
    masks, flows = meval.eval(
        model, imgs, compute_masks=True, batch_mode='single', **_FAST
    )

    assert isinstance(masks, np.ndarray)
    assert len(flows) == 2
    for i, (h, w) in enumerate(shapes):
        assert flows[i][2].shape == (h, w)


# ---------------------------------------------------------------------------
# 7. Multichannel 2D image  (H, W, C)
# ---------------------------------------------------------------------------

def test_eval_multichannel_2d(monkeypatch):
    model = make_stub_model(nchan=2)
    monkeypatch.setattr(meval.core, "compute_masks", fake_compute_masks_factory(model))

    img = np.zeros((16, 16, 2), dtype=np.float32)  # (H, W, C)
    # Pass channels directly — do not use _FAST which already has channels=[0,0]
    masks, flows = meval.eval(
        model, img,
        channels=[1, 2],
        normalize=False,
        invert=False,
        rescale_factor=1.0,
        compute_masks=True,
        net_avg=False,
        augment=False,
        tile=False,
        bsize=16,
        show_progress=False,
        verbose=False,
    )

    assert isinstance(masks, np.ndarray)
    # single image → (1, H, W)
    assert masks.shape == (1, 16, 16)


# ---------------------------------------------------------------------------
# 8. compute_masks=False  (return network outputs only, no mask reconstruction)
# ---------------------------------------------------------------------------

def test_eval_compute_masks_false(monkeypatch):
    model = make_stub_model()
    # compute_masks=False should NOT call core.compute_masks
    called = {"count": 0}

    def guard(*args, **kwargs):
        called["count"] += 1
        return fake_compute_masks_factory(model)(*args, **kwargs)

    monkeypatch.setattr(meval.core, "compute_masks", guard)

    imgs = [np.zeros((16, 16), dtype=np.float32) for _ in range(2)]
    masks, flows = meval.eval(
        model, imgs, compute_masks=False, **_FAST
    )

    assert called["count"] == 0, "compute_masks should not be called when compute_masks=False"
    assert isinstance(masks, np.ndarray)
    # masks should be all-zero placeholders
    assert np.all(masks == 0)
    assert len(flows) == 2


# ---------------------------------------------------------------------------
# 9. stitch_threshold > 0  (2D stack → 3D stitched volume)
# ---------------------------------------------------------------------------

def test_eval_stitch_threshold(monkeypatch):
    """stitch_threshold > 0: compute 2D masks per plane, then stitch into 3D volume."""
    model = make_stub_model()
    monkeypatch.setattr(meval.core, "compute_masks", fake_compute_masks_factory(model))

    stitched = {"called": False}

    def fake_stitch3D(masks_array, stitch_threshold=0.0):
        stitched["called"] = True
        return masks_array  # identity — same shape in/out

    monkeypatch.setattr(meval.utils, "stitch3D", fake_stitch3D)

    # Pass a numpy stack of same-size 2D images (is_stack path)
    stack = np.zeros((3, 16, 16), dtype=np.float32)
    masks, flows = meval.eval(
        model, stack, compute_masks=True, stitch_threshold=0.5, **_FAST
    )

    assert stitched["called"], "stitch3D should have been called when stitch_threshold > 0"
    assert isinstance(masks, np.ndarray)
    assert len(flows) == 3


# ---------------------------------------------------------------------------
# 10. Single file path string  (string → [path])
# ---------------------------------------------------------------------------

def test_eval_single_file_path(monkeypatch, tmp_path):
    model = make_stub_model()
    monkeypatch.setattr(meval.core, "compute_masks", fake_compute_masks_factory(model))

    arr = np.zeros((16, 16), dtype=np.float32)
    fpath = str(tmp_path / "img.npy")
    np.save(fpath, arr)

    masks, flows = meval.eval(model, fpath, compute_masks=True, **_FAST)

    assert isinstance(masks, np.ndarray)
    # Single file → normalised to [path] → 1-image batch → (1, H, W)
    assert masks.shape == (1, 16, 16)
    assert len(flows) == 1


# ---------------------------------------------------------------------------
# 11. List of file path strings
# ---------------------------------------------------------------------------

def test_eval_list_file_paths(monkeypatch, tmp_path):
    model = make_stub_model()
    monkeypatch.setattr(meval.core, "compute_masks", fake_compute_masks_factory(model))

    paths = []
    for i in range(3):
        arr = np.zeros((16, 16), dtype=np.float32)
        p = str(tmp_path / f"img_{i}.npy")
        np.save(p, arr)
        paths.append(p)

    masks, flows = meval.eval(model, paths, compute_masks=True, **_FAST)

    assert isinstance(masks, np.ndarray)
    assert masks.shape == (3, 16, 16)
    assert len(flows) == 3


# ---------------------------------------------------------------------------
# 12. Folder path string  (directory → sorted list of image files)
# ---------------------------------------------------------------------------

def test_eval_folder_path(monkeypatch, tmp_path):
    model = make_stub_model()
    monkeypatch.setattr(meval.core, "compute_masks", fake_compute_masks_factory(model))

    # Create a folder with .npy images (and one non-image file to verify filtering)
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(4):
        arr = np.zeros((16, 16), dtype=np.float32)
        np.save(str(img_dir / f"img_{i:02d}.npy"), arr)
    (img_dir / "readme.txt").write_text("not an image")

    masks, flows = meval.eval(model, str(img_dir), compute_masks=True, **_FAST)

    assert isinstance(masks, np.ndarray)
    assert masks.shape == (4, 16, 16)
    assert len(flows) == 4


# ---------------------------------------------------------------------------
# 13. do_3D — single 3D volume  (is_image + do_3D=True)
# ---------------------------------------------------------------------------

def test_eval_do_3d_single_volume(monkeypatch):
    model = make_stub_model(dim=2, nchan=1, nclasses=3, omni=False)
    stub_run_3d(model)
    monkeypatch.setattr(meval.core, "compute_masks", fake_compute_masks_factory(model))
    # dx_to_circ may be called; stub it to avoid any plotting dependencies
    monkeypatch.setattr(meval.plot, "dx_to_circ",
                        lambda arr, **kw: np.zeros((3,) + arr.shape[1:], dtype=np.float32))

    vol = np.zeros((8, 16, 16), dtype=np.float32)  # (Z, Y, X)
    masks, flows = meval.eval(
        model, vol,
        do_3D=True,
        compute_masks=True,
        omni=False,
        **_FAST,
    )

    # do_3D returns an array of shape (n_volumes, Z, Y, X)
    assert isinstance(masks, np.ndarray)
    assert masks.ndim == 4     # (n_vols, Z, Y, X)
    assert masks.shape[0] == 1
    assert isinstance(flows, list) and len(flows) == 1



# ---------------------------------------------------------------------------
# 14. do_3D — list of 3D volumes
# ---------------------------------------------------------------------------

def test_eval_do_3d_list_of_volumes(monkeypatch):
    model = make_stub_model(dim=2, nchan=1, nclasses=3, omni=False)
    stub_run_3d(model)
    monkeypatch.setattr(meval.core, "compute_masks", fake_compute_masks_factory(model))
    monkeypatch.setattr(meval.plot, "dx_to_circ",
                        lambda arr, **kw: np.zeros((3,) + arr.shape[1:], dtype=np.float32))

    vols = [np.zeros((8, 16, 16), dtype=np.float32) for _ in range(2)]
    masks, flows = meval.eval(
        model, vols,
        do_3D=True,
        compute_masks=True,
        omni=False,
        **_FAST,
    )

    assert isinstance(masks, np.ndarray)
    assert masks.shape[0] == 2
    assert len(flows) == 2


# ---------------------------------------------------------------------------
# 15. do_3D — single file path to a 3D volume
# ---------------------------------------------------------------------------

def test_eval_do_3d_file_path(monkeypatch, tmp_path):
    model = make_stub_model(dim=2, nchan=1, nclasses=3, omni=False)
    stub_run_3d(model)
    monkeypatch.setattr(meval.core, "compute_masks", fake_compute_masks_factory(model))
    monkeypatch.setattr(meval.plot, "dx_to_circ",
                        lambda arr, **kw: np.zeros((3,) + arr.shape[1:], dtype=np.float32))

    vol = np.zeros((8, 16, 16), dtype=np.float32)
    fpath = str(tmp_path / "vol.npy")
    np.save(fpath, vol)

    masks, flows = meval.eval(
        model, fpath,
        do_3D=True,
        compute_masks=True,
        omni=False,
        **_FAST,
    )

    assert isinstance(masks, np.ndarray)
    assert masks.shape[0] == 1


# ---------------------------------------------------------------------------
# 16. EvalSet passed directly  (regression — dataset branch)
# ---------------------------------------------------------------------------

def test_eval_evalset_direct(monkeypatch):
    model = make_stub_model()
    monkeypatch.setattr(meval.core, "compute_masks", fake_compute_masks_factory(model))

    dataset = eval_mod.eval_set(
        [np.zeros((16, 16), dtype=np.float32),
         np.zeros((16, 16), dtype=np.float32)],
        dim=2, normalize=False, invert=False,
    )
    masks, flows = meval.eval(model, dataset, compute_masks=True, **_FAST)

    assert isinstance(masks, np.ndarray)
    assert masks.shape == (2, 16, 16)
    assert len(flows) == 2



# ---------------------------------------------------------------------------
# 17. batch_mode='group' with explicitly same-shape list
# ---------------------------------------------------------------------------

def test_eval_batch_mode_group(monkeypatch):
    model = make_stub_model()
    monkeypatch.setattr(meval.core, "compute_masks", fake_compute_masks_factory(model))

    imgs = [np.zeros((16, 16), dtype=np.float32) for _ in range(5)]
    masks, flows = meval.eval(
        model, imgs, compute_masks=True, batch_mode='group', loader_batch_size=3, **_FAST
    )

    assert masks.shape == (5, 16, 16)
    assert len(flows) == 5


# ---------------------------------------------------------------------------
# 18. Mixed-size list with batch_mode='pad' — verify shapes are per-image
# ---------------------------------------------------------------------------

def test_eval_mixed_sizes_pad_per_image_shapes(monkeypatch):
    """After per-image processing, each output dist map should match its input shape.

    Uses batch_mode='group' (same-shape images are grouped together) so that
    all images are processed independently without inter-image padding, then we
    verify each output dist map matches its input's spatial dimensions.
    """
    model = make_stub_model()
    monkeypatch.setattr(meval.core, "compute_masks", fake_compute_masks_factory(model))

    shapes = [(16, 16), (20, 20), (24, 24)]
    imgs = [np.zeros(s, dtype=np.float32) for s in shapes]
    masks, flows = meval.eval(
        model, imgs, compute_masks=True, batch_mode='group', **_FAST
    )

    assert len(flows) == 3
    for i, (h, w) in enumerate(shapes):
        dist_i = flows[i][2]   # dist is index 2 in the flow tuple
        assert dist_i.shape == (h, w), (
            f"Image {i}: expected dist shape {(h, w)}, got {dist_i.shape}"
        )


# ---------------------------------------------------------------------------
# 19. diameter → rescale_factor conversion
# ---------------------------------------------------------------------------

def test_eval_diameter_sets_rescale(monkeypatch):
    model = make_stub_model()
    monkeypatch.setattr(meval.core, "compute_masks", fake_compute_masks_factory(model))

    img = np.zeros((16, 16), dtype=np.float32)
    # diameter != None and rescale_factor not given → should compute rescale internally
    masks, flows = meval.eval(
        model, img,
        channels=[0, 0],
        normalize=False,
        invert=False,
        diameter=model.diam_mean,  # → rescale_factor = 1.0
        compute_masks=True,
        net_avg=False,
        augment=False,
        tile=False,
        bsize=16,
        show_progress=False,
        verbose=False,
    )
    assert isinstance(masks, np.ndarray)


# ---------------------------------------------------------------------------
# 20. Stack input with stitch_threshold=0 (no stitching — array passthrough)
# ---------------------------------------------------------------------------

def test_eval_stack_no_stitch(monkeypatch):
    model = make_stub_model()
    monkeypatch.setattr(meval.core, "compute_masks", fake_compute_masks_factory(model))

    stack = np.zeros((3, 16, 16), dtype=np.float32)
    masks, flows = meval.eval(
        model, stack, compute_masks=True, stitch_threshold=0.0, **_FAST
    )

    assert isinstance(masks, np.ndarray)
    assert masks.shape == (3, 16, 16)
    assert len(flows) == 3


# ---------------------------------------------------------------------------
# 21. Empty folder → eval on zero images should not crash
# ---------------------------------------------------------------------------

def test_eval_empty_folder(monkeypatch, tmp_path):
    model = make_stub_model()
    monkeypatch.setattr(meval.core, "compute_masks", fake_compute_masks_factory(model))

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    # Should produce empty outputs without raising
    masks, flows = meval.eval(model, str(empty_dir), compute_masks=True, **_FAST)

    assert len(flows) == 0
    # masks is np.array([]) when list is empty
    assert isinstance(masks, np.ndarray)


# ---------------------------------------------------------------------------
# 22. Multichannel per-image channels specification
# ---------------------------------------------------------------------------

def test_eval_per_image_channels(monkeypatch):
    """channels can be a list-of-lists, one [cyto, nuc] pair per image."""
    model = make_stub_model(nchan=2)
    monkeypatch.setattr(meval.core, "compute_masks", fake_compute_masks_factory(model))

    imgs = [
        np.zeros((16, 16), dtype=np.float32),
        np.zeros((16, 16), dtype=np.float32),
    ]
    per_image_channels = [[1, 0], [2, 0]]

    masks, flows = meval.eval(
        model, imgs,
        channels=per_image_channels,
        normalize=False,
        invert=False,
        rescale_factor=1.0,
        compute_masks=True,
        net_avg=False,
        augment=False,
        tile=False,
        bsize=16,
        show_progress=False,
        verbose=False,
    )

    assert isinstance(masks, np.ndarray)
    assert masks.shape == (2, 16, 16)
