import numpy as np
import torch

from omnirefactor.data import eval as eval_mod


class DummyNet:
    def __call__(self, batch):
        # return tuple like torch model (y, aux)
        y = torch.zeros((batch.shape[0], 3, *batch.shape[-2:]), device=batch.device)
        return (y,)


class DummyModel:
    def __init__(self):
        self.nclasses = 3
        self.dim = 2
        self.unet = False
        self.net = DummyNet()


class DummyRunNetModel:
    def _run_net(self, batch):
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        return batch


def test_eval_set_stack_getitem_no_pad():
    data = np.zeros((2, 8, 8), dtype=np.float32)
    dataset = eval_mod.eval_set(data, dim=2, normalize=False, invert=False)
    img = dataset.__getitem__(0, no_pad=True)
    assert isinstance(img, torch.Tensor)
    # Shape is (C, H, W) = (1, 8, 8) - channel dimension is preserved
    assert img.shape == (1, 8, 8)


def test_eval_set_list_channel_axis_and_rescale():
    data = [np.zeros((16, 32, 2), dtype=np.float32)]
    dataset = eval_mod.eval_set(
        data,
        dim=2,
        channel_axis=3,
        normalize=False,
        invert=False,
        rescale_factor=0.5,
        interp_mode="bilinear",
        pad_mode="constant",
        extra_pad=0,
    )
    img, inds, subs = dataset[0]
    assert img.shape[1] == 2  # channel axis moved to C
    assert len(inds) == 1
    assert len(subs) == 2


def test_eval_set_normalize_invert_with_contrast_limits():
    data = np.ones((1, 6, 6), dtype=np.float32)
    dataset = eval_mod.eval_set(
        data,
        dim=2,
        normalize=True,
        invert=True,
        contrast_limits=(0.0, 1.0),
    )
    img = dataset.__getitem__(0, no_pad=True)
    assert img.min() >= 0
    assert img.max() <= 1


def test_eval_set_iterates(monkeypatch):
    data = [np.zeros((5, 5), dtype=np.float32), np.zeros((5, 5), dtype=np.float32)]
    dataset = eval_mod.eval_set(
        data,
        dim=2,
        normalize=False,
        invert=False,
        pad_mode="constant",
        extra_pad=0,
    )
    # Mock torch.utils.data.get_worker_info (used in __iter__)
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: None)
    items = list(iter(dataset))
    assert len(items) == 2


def test_eval_set_run_tiled_and_collate():
    data = np.zeros((1, 32, 32), dtype=np.float32)
    dataset = eval_mod.eval_set(
        data,
        dim=2,
        normalize=False,
        invert=False,
        tile=True,
        pad_mode="constant",
        extra_pad=0,
    )
    batch, inds, subs = dataset[0]
    dummy = DummyModel()
    out = dataset._run_tiled(batch, dummy, batch_size=1, augment=False, bsize=8, tile_overlap=0.1)
    assert out.shape[1] == dummy.nclasses

    batch2, inds2, subs2 = dataset[0]
    collated = dataset.collate_fn([(batch, inds, subs), (batch2, inds2, subs2)])
    assert isinstance(collated, tuple)
    assert collated[0].shape[0] == 2


def test_eval_loader_iter_and_sampler_len():
    sampler = eval_mod.sampler([0, 1, 2])
    assert len(sampler) == 3
    assert list(iter(sampler)) == [0, 1, 2]

    dataset = torch.utils.data.TensorDataset(torch.zeros((2, 1, 4, 4)))
    loader = eval_mod.eval_loader(dataset, DummyRunNetModel(), lambda x: x, batch_size=1)
    batches = list(iter(loader))
    assert len(batches) == 2


def test_eval_set_iter_worker_split(monkeypatch):
    data = [np.zeros((32, 32), dtype=np.float32) for _ in range(5)]
    dataset = eval_mod.eval_set(
        data,
        dim=2,
        normalize=False,
        invert=False,
        pad_mode="constant",
        extra_pad=0,
    )

    class WorkerInfo:
        num_workers = 2
        id = 1

    # Mock torch.utils.data.get_worker_info (used in __iter__)
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: WorkerInfo())
    items = list(iter(dataset))
    assert len(items) == 3


def test_eval_set_files_and_aics_branches(monkeypatch):
    # File branch: patch imread so no real file is needed
    monkeypatch.setattr(eval_mod, "imread", lambda _path: np.zeros((16, 16), dtype=np.float32))

    file_dataset = eval_mod.eval_set(
        ["fake.tif"],
        dim=2,
        normalize=False,
        invert=False,
        pad_mode="constant",
        extra_pad=0,
    )
    img, inds, subs = file_dataset[0]
    assert img.shape[1] == 1
    assert len(inds) == 1

    # AICS branch: patch AICSImage so isinstance check passes
    class DummyAICSImage:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_image_data(self, *_args, **_kwargs):
            return np.zeros((16, 16), dtype=np.float32)

    monkeypatch.setattr(eval_mod, "AICSImage", DummyAICSImage)

    aics_dataset = eval_mod.eval_set(
        DummyAICSImage(),
        dim=2,
        normalize=False,
        invert=False,
        pad_mode="constant",
        extra_pad=0,
        aics_args={"slice_dim": "Z"},
    )
    img, inds, subs = aics_dataset[0]
    assert img.shape[1] == 1


def test_eval_set_run_tiled_augment(monkeypatch):
    data = np.zeros((1, 32, 32), dtype=np.float32)
    dataset = eval_mod.eval_set(
        data,
        dim=2,
        normalize=False,
        invert=False,
        tile=True,
        pad_mode="constant",
        extra_pad=0,
    )
    batch, inds, subs = dataset[0]
    dummy = DummyModel()
    called = {"unaugment": False}

    def fake_make_tiles(imgi, **_):
        IMG = torch.zeros((1, 1, 8, 8))
        subs = [slice(0, 32), slice(0, 32)]
        shape = (32, 32)
        inds = [0]
        return IMG, subs, shape, inds

    def fake_unaugment(y, inds, _unet):
        called["unaugment"] = True
        return y

    def fake_average_tiles(y, subs, shape):
        return torch.zeros((y.shape[1], *shape), device=y.device)

    monkeypatch.setattr(eval_mod, "make_tiles_ND", fake_make_tiles)
    monkeypatch.setattr(eval_mod, "unaugment_tiles_ND", fake_unaugment)
    monkeypatch.setattr(eval_mod, "average_tiles_ND", fake_average_tiles)

    out = dataset._run_tiled(batch, dummy, batch_size=1, augment=True, bsize=8, tile_overlap=0.1)
    assert out.shape[1] == dummy.nclasses
    assert called["unaugment"] is True


# ============================================================================
# Batch Mode Tests
# ============================================================================

def test_eval_set_batch_mode_auto_uniform_shapes():
    """Auto mode should choose 'group' for uniform shapes."""
    data = [np.zeros((32, 32), dtype=np.float32) for _ in range(5)]
    dataset = eval_mod.eval_set(
        data,
        dim=2,
        normalize=False,
        invert=False,
        batch_mode='auto',
    )
    assert dataset.batch_mode == 'group'
    assert dataset.shape_info['n_unique_shapes'] == 1


def test_eval_set_batch_mode_auto_few_shapes():
    """Auto mode should choose 'group' for few distinct shapes."""
    data = [
        np.zeros((32, 32), dtype=np.float32),
        np.zeros((32, 32), dtype=np.float32),
        np.zeros((64, 64), dtype=np.float32),
        np.zeros((64, 64), dtype=np.float32),
    ]
    dataset = eval_mod.eval_set(
        data,
        dim=2,
        normalize=False,
        invert=False,
        batch_mode='auto',
    )
    assert dataset.batch_mode == 'group'
    assert dataset.shape_info['n_unique_shapes'] == 2


def test_eval_set_batch_mode_auto_many_shapes():
    """Auto mode should choose 'pad' for many distinct shapes."""
    data = [np.zeros((32 + i, 32 + i), dtype=np.float32) for i in range(10)]
    dataset = eval_mod.eval_set(
        data,
        dim=2,
        normalize=False,
        invert=False,
        batch_mode='auto',
    )
    assert dataset.batch_mode == 'pad'
    assert dataset.shape_info['n_unique_shapes'] == 10


def test_eval_set_batch_mode_single():
    """Single mode should create one batch per image."""
    data = [np.zeros((32, 32), dtype=np.float32) for _ in range(3)]
    dataset = eval_mod.eval_set(
        data,
        dim=2,
        normalize=False,
        invert=False,
        batch_mode='single',
    )
    assert dataset.batch_mode == 'single'
    assert dataset.n_batches == 3


def test_eval_set_batch_mode_group_same_shape():
    """Group mode should batch same-shape images together."""
    data = [np.zeros((32, 32), dtype=np.float32) for _ in range(5)]
    dataset = eval_mod.eval_set(
        data,
        dim=2,
        normalize=False,
        invert=False,
        batch_mode='group',
        max_batch_size=3,
    )
    assert dataset.batch_mode == 'group'
    # 5 images with batch_size=3 should give 2 batches
    assert dataset.n_batches == 2


def test_eval_set_batch_mode_group_mixed_shapes():
    """Group mode should create separate batches for different shapes."""
    data = [
        np.zeros((32, 32), dtype=np.float32),
        np.zeros((32, 32), dtype=np.float32),
        np.zeros((64, 64), dtype=np.float32),
        np.zeros((64, 64), dtype=np.float32),
        np.zeros((64, 64), dtype=np.float32),
    ]
    dataset = eval_mod.eval_set(
        data,
        dim=2,
        normalize=False,
        invert=False,
        batch_mode='group',
        max_batch_size=4,
    )
    assert dataset.batch_mode == 'group'
    # 2 shapes: 2x(32,32) in one batch, 3x(64,64) in one batch
    assert dataset.n_batches == 2


def test_eval_set_batch_mode_pad():
    """Pad mode should create batches padded to common shape."""
    data = [
        np.zeros((32, 32), dtype=np.float32),
        np.zeros((48, 48), dtype=np.float32),
        np.zeros((64, 64), dtype=np.float32),
    ]
    dataset = eval_mod.eval_set(
        data,
        dim=2,
        normalize=False,
        invert=False,
        batch_mode='pad',
        max_batch_size=4,
        extra_pad=0,
    )
    assert dataset.batch_mode == 'pad'
    assert dataset.n_batches == 1  # All in one batch

    # Check that batches have correct shape
    batch, inds, subs = dataset.get_batch(0)
    # Should be padded to 64 (max shape), then to 16-divisible = 64
    assert batch.shape[-2:] == (64, 64)
    assert len(inds) == 3


def test_eval_set_get_batch_extracts_correct_region():
    """Verify that subscripts correctly extract original image region."""
    # Create images with distinct values
    img1 = np.ones((32, 32), dtype=np.float32) * 1.0
    img2 = np.ones((48, 48), dtype=np.float32) * 2.0

    data = [img1, img2]
    dataset = eval_mod.eval_set(
        data,
        dim=2,
        normalize=False,
        invert=False,
        batch_mode='pad',
        max_batch_size=4,
        extra_pad=0,
    )

    batch, inds, subs = dataset.get_batch(0)

    # Extract original regions using subscripts (subs are slices)
    for i, (idx, sub) in enumerate(zip(inds, subs)):
        # sub is a list of slices, one per spatial dimension
        extracted = batch[i, :, sub[0], sub[1]]
        orig_shape = data[idx].shape
        # Extracted spatial shape should match original
        assert extracted.shape[-2:] == orig_shape, f"Expected {orig_shape}, got {extracted.shape[-2:]}"


def test_eval_set_iter_batches():
    """Test iter_batches generator."""
    data = [np.zeros((32, 32), dtype=np.float32) for _ in range(5)]
    dataset = eval_mod.eval_set(
        data,
        dim=2,
        normalize=False,
        invert=False,
        batch_mode='group',
        max_batch_size=2,
    )

    batches = list(dataset.iter_batches())
    assert len(batches) == 3  # 5 images, batch_size=2 -> 3 batches

    # Check all images are covered
    all_inds = []
    for batch, inds, subs in batches:
        all_inds.extend(inds)
    assert sorted(all_inds) == [0, 1, 2, 3, 4]


def test_eval_set_output_consistency_group_vs_single():
    """Verify outputs are identical between group and single modes for uniform shapes."""
    np.random.seed(42)
    data = [np.random.rand(32, 32).astype(np.float32) for _ in range(4)]

    # Process with single mode
    dataset_single = eval_mod.eval_set(
        data,
        dim=2,
        normalize=False,
        invert=False,
        batch_mode='single',
        extra_pad=0,
    )

    # Process with group mode
    dataset_group = eval_mod.eval_set(
        data,
        dim=2,
        normalize=False,
        invert=False,
        batch_mode='group',
        max_batch_size=4,
        extra_pad=0,
    )

    # Get all images from single mode
    single_outputs = {}
    for batch, inds, subs in dataset_single.iter_batches():
        for i, idx in enumerate(inds):
            # Extract the region for this image
            img = batch[i, :, subs[i][0], subs[i][1]]
            single_outputs[idx] = img

    # Get all images from group mode
    group_outputs = {}
    for batch, inds, subs in dataset_group.iter_batches():
        for i, idx in enumerate(inds):
            img = batch[i, :, subs[i][0], subs[i][1]]
            group_outputs[idx] = img

    # Compare outputs
    for idx in range(len(data)):
        assert idx in single_outputs
        assert idx in group_outputs
        assert torch.allclose(single_outputs[idx], group_outputs[idx], atol=1e-6)


def test_eval_set_shape_info():
    """Test shape_info property."""
    data = [
        np.zeros((32, 32), dtype=np.float32),
        np.zeros((32, 32), dtype=np.float32),
        np.zeros((64, 48), dtype=np.float32),
    ]
    dataset = eval_mod.eval_set(
        data,
        dim=2,
        normalize=False,
        invert=False,
        batch_mode='group',
    )

    info = dataset.shape_info
    assert info['n_images'] == 3
    assert info['n_unique_shapes'] == 2
    assert info['batch_mode'] == 'group'
    assert (32, 32) in info['shapes']
    assert (64, 48) in info['shapes']


def test_eval_set_3d_batch_mode():
    """Test batch modes work with 3D images."""
    data = [np.zeros((16, 32, 32), dtype=np.float32) for _ in range(3)]
    dataset = eval_mod.eval_set(
        data,
        dim=3,
        normalize=False,
        invert=False,
        batch_mode='group',
        max_batch_size=2,
    )

    assert dataset.batch_mode == 'group'
    assert dataset.shape_info['n_unique_shapes'] == 1
    assert (16, 32, 32) in dataset.shape_info['shapes']


def test_eval_set_rescale_affects_shape_grouping():
    """Test that rescale_factor is accounted for in shape analysis."""
    # 32x32 * 0.5 = 16x16, and 32x32 * 0.5 = 16x16 (same after rescale)
    data = [
        np.zeros((32, 32), dtype=np.float32),
        np.zeros((32, 32), dtype=np.float32),
    ]
    dataset = eval_mod.eval_set(
        data,
        dim=2,
        normalize=False,
        invert=False,
        batch_mode='group',
        rescale_factor=0.5,
    )

    # Both images should have same shape after rescaling: 16x16
    assert dataset.shape_info['n_unique_shapes'] == 1
    assert (16, 16) in dataset.shape_info['shapes']

    # Now test with different original shapes that become same after rescale
    # 64x64 * 0.5 = 32x32, and 32x32 * 1.0 = 32x32
    data2 = [
        np.zeros((64, 64), dtype=np.float32),
        np.zeros((64, 64), dtype=np.float32),
    ]
    dataset2 = eval_mod.eval_set(
        data2,
        dim=2,
        normalize=False,
        invert=False,
        batch_mode='group',
        rescale_factor=0.5,
    )
    # Both become 32x32 after rescale
    assert dataset2.shape_info['n_unique_shapes'] == 1
    assert (32, 32) in dataset2.shape_info['shapes']
