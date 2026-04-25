import numpy as np
import torch

from omnipose.utils import neighbor


def test_fields_torchscript_python_paths():
    # Exercise eikonal_update_torch -> update_torch path.
    import omnipose.core.fields as fields
    Tneigh = torch.tensor([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]])
    index_list = [torch.tensor([0, 1]), torch.tensor([0, 1])]
    factors = torch.tensor([0.0, 1.0])
    fn = fields.eikonal_update_torch
    for attr in ("_orig_mod", "_torchdynamo_orig_callable", "__wrapped__"):
        if hasattr(fn, attr):
            fn = getattr(fn, attr)
            break
    _ = fn(
        Tneigh,
        torch.tensor([0, 1, 2]),
        torch.tensor(2),
        index_list,
        factors,
    )

    # Exercise _iterate with omni=True (hits eikonal_update_torch and smoothing path).
    T = torch.ones(2)
    neigh_inds = torch.tensor([[0, 1], [1, 0]])
    central_inds = torch.tensor([0, 1], dtype=torch.long)
    centroid_inds = torch.tensor([], dtype=torch.long)
    _ = fields._iterate(
        T,
        neigh_inds,
        central_inds,
        centroid_inds,
        torch.tensor(0),
        torch.tensor(2),
        [torch.tensor([0, 1]), torch.tensor([0, 1])],
        torch.tensor([0.0, 1.0]),
        torch.ones_like(neigh_inds, dtype=torch.bool),
        torch.tensor(1),
        torch.tensor(True),
        torch.tensor(False),
    )

    # Exercise _iterate with omni=False branch.
    _ = fields._iterate(
        T,
        neigh_inds,
        central_inds,
        centroid_inds,
        torch.tensor(0),
        torch.tensor(2),
        [torch.tensor([0, 1]), torch.tensor([0, 1])],
        torch.tensor([0.0, 1.0]),
        torch.ones_like(neigh_inds, dtype=torch.bool),
        torch.tensor(1),
        torch.tensor(False),
        torch.tensor(False),
    )

    # Exercise _gradient with a small, consistent neighborhood setup.
    mask = np.zeros((3, 3), dtype=np.uint8)
    mask[1, 1] = 1
    mask[1, 2] = 1
    coords = np.nonzero(mask)
    steps, inds, idx, fact, _sign = neighbor.kernel_setup(mask.ndim)
    neighbors = neighbor.get_neighbors(coords, steps, mask.ndim, mask.shape)
    _indexes, neigh_inds, _ind_matrix = neighbor.get_neigh_inds(neighbors, coords, mask.shape, background_reflect=True)

    T = torch.tensor([1.0, 2.0], dtype=torch.float32)
    steps_t = torch.tensor(steps)
    inds_t = [torch.tensor(i) for i in inds]
    fact_t = torch.tensor(fact)
    neigh_inds_t = torch.tensor(neigh_inds)
    central_inds_t = torch.tensor([0, 1], dtype=torch.long)
    isneigh = torch.ones_like(neigh_inds_t, dtype=torch.bool)
    n_axes = len(fact) - 1
    s = [n_axes, mask.ndim, neigh_inds.shape[-1]]

    _ = fields._gradient(
        T,
        torch.tensor(mask.ndim),
        steps_t,
        fact_t,
        inds_t,
        isneigh,
        neigh_inds_t,
        central_inds_t,
        s,
    )


def test_update_torch_nontrivial():
    import omnipose.core.fields as fields
    a = torch.tensor([[0.1, 0.5], [0.4, 0.6], [0.9, 1.2]])
    f = torch.tensor(0.6)
    fsq = torch.tensor(0.36)
    out = fields.update_torch(a, f, fsq)
    assert out.shape == a.shape[1:]


def test_divergence_small_dims():
    import omnipose.core.fields as fields
    f = np.zeros((2, 1, 3), dtype=np.float32)
    out = fields.divergence(f)
    assert out.shape == f[0].shape


def test_divergence_torch_small_dims():
    import omnipose.core.fields as fields
    y = torch.zeros((1, 2, 1, 3))
    out = fields.divergence_torch(y)
    assert out.shape == (1, 1, 3)


def test_divergence_torch_mps():
    import omnipose.core.fields as fields
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return
    y = torch.randn((1, 2, 4, 4), device="mps")
    out = fields.divergence_torch(y)
    assert out.device.type == "mps"
