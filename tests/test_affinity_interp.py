"""Regression tests for the ND flow-following interpolation used by AffinityLoss.

Guards against the torchvf ``nearest_interpolation_batched`` bug where
``vf.gather(-1, points)`` only indexed the last spatial axis, so in ND only 1
of D flow components was sampled correctly (streaked lossE/lossA in 2D and 3D).

The decisive correctness guarantee here is *axis-permutation / reflection
invariance* of the whole AffinityLoss: it is a coordinate-free scalar, so
relabelling or flipping the spatial axes must not change it. The old bug
privileged the last axis and violated this; the fixed interpolator restores it.
"""
import itertools

import numpy as np
import torch

from omnipose.metrics.loss import (
    nearest_interp_batched,
    ivp_euler_batched,
    AffinityLoss,
)


def _distinctive_field(shape):
    """Field whose value encodes (component, ND-coordinate) so a wrong
    sample location is detectable."""
    dim = len(shape)
    coords = torch.meshgrid(*[torch.arange(s) for s in shape], indexing="ij")
    return torch.stack(
        [(c + 1) * 1000.0 + sum(coords[k] * (10 ** k) for k in range(dim))
         for c in range(dim)], 0
    )[None].float()


def _identity_grid(shape):
    coords = torch.meshgrid(*[torch.arange(s) for s in shape], indexing="ij")
    return torch.stack(coords, 0)[None].float()


def test_identity_sampling_returns_field_2d():
    shape = (5, 7)
    vf = _distinctive_field(shape)
    out = nearest_interp_batched(vf, _identity_grid(shape))
    assert torch.equal(out, vf), "sampling at identity grid must return the field"


def test_identity_sampling_returns_field_3d():
    shape = (4, 5, 6)
    vf = _distinctive_field(shape)
    out = nearest_interp_batched(vf, _identity_grid(shape))
    assert torch.equal(out, vf)


def test_all_components_sampled_correctly():
    # the old bug left only the LAST component correct; assert every component is
    for shape in [(6, 8), (4, 6, 8)]:
        dim = len(shape)
        vf = _distinctive_field(shape)
        out = nearest_interp_batched(vf, _identity_grid(shape))
        for c in range(dim):
            err = (out[0, c] - vf[0, c]).abs().max().item()
            assert err == 0.0, f"component {c} mis-sampled (dim={dim})"


def test_matches_bruteforce_random_points():
    torch.manual_seed(0)
    for shape in [(6, 9), (4, 5, 7)]:
        B, D = 2, len(shape)
        vf = torch.randn(B, D, *shape)
        pts = torch.rand(B, D, *shape) * (max(shape) + 3) - 2  # spans out-of-range too
        out = nearest_interp_batched(vf, pts)
        idx = [torch.clamp(pts[:, k], 0, shape[k] - 1).round().long() for k in range(D)]
        ref = torch.empty_like(vf)
        for b in range(B):
            for c in range(D):
                ref[b, c] = vf[b, c][tuple(idx[k][b] for k in range(D))]
        assert torch.equal(out, ref)


def test_euler_zero_field_is_stationary():
    shape = (5, 7)
    vf = torch.zeros(1, 2, *shape)
    init = _identity_grid(shape)
    out = ivp_euler_batched(vf, init, dx=0.5, n_steps=3)
    assert torch.equal(out, init)


def test_euler_constant_flow_is_exact():
    # For a spatially-constant flow the sampled value is v everywhere (even after
    # clamping), so after n steps every point is exactly init + n*dx*v.
    for shape in [(9, 11), (7, 8, 9)]:
        dim = len(shape)
        v = torch.arange(1, dim + 1, dtype=torch.float32)
        vf = v.reshape(1, dim, *([1] * dim)).expand(1, dim, *shape).clone()
        init = _identity_grid(shape)
        out = ivp_euler_batched(vf, init, dx=0.25, n_steps=4)
        expected = init + 4 * 0.25 * v.reshape(1, dim, *([1] * dim))
        assert torch.allclose(out, expected, atol=1e-5)


def test_euler_matches_independent_bruteforce():
    # Independent integrator: python-indexed sampling, same Euler recurrence.
    torch.manual_seed(1)
    for shape in [(8, 10), (6, 7, 8)]:
        dim = len(shape)
        vf = torch.randn(1, dim, *shape)
        pts = _identity_grid(shape)
        ref = pts.clone()
        for _ in range(3):
            idx = [torch.clamp(ref[:, k], 0, shape[k] - 1).round().long() for k in range(dim)]
            samp = torch.stack([vf[0, c][tuple(idx[k][0] for k in range(dim))]
                                for c in range(dim)], 0)[None]
            ref = ref + 0.3 * samp
        out = ivp_euler_batched(vf, pts, dx=0.3, n_steps=3)
        assert torch.allclose(out, ref, atol=1e-5)


# --- axis-permutation / reflection equivariance of the full AffinityLoss -------
def _permute_scalar(S, perm):
    return S.permute(0, *[1 + perm[i] for i in range(len(perm))]).contiguous()


def _permute_vector(F, perm):
    D = len(perm)
    F_sp = F.permute(0, 1, *[2 + perm[i] for i in range(D)])
    return F_sp[:, list(perm)].contiguous()   # component i <- old component perm[i]


def _asymmetric_pred_gt(dim, device):
    """gt = radial-inward flow; pred = same but per-axis scaled so the config is
    NOT axis-symmetric -> a loss that privileges any axis will change under a
    permutation."""
    flow, dist = _make_flow_dist(dim, device)
    scale = torch.tensor([1.0 + 0.4 * k for k in range(dim)]).reshape(1, dim, *([1] * dim))
    return flow * scale, flow, dist


def test_affinity_loss_axis_permutation_invariant():
    for dim in (2, 3):
        device = torch.device("cpu")
        crit = AffinityLoss(device, dim)
        flow_pred, flow_gt, dist = _asymmetric_pred_gt(dim, device)
        base = [float(x) for x in crit(flow_pred, dist, flow_gt, dist, mode="all")]
        assert max(base[:2]) > 1e-4, "need a nonzero loss for the test to have teeth"
        for perm in itertools.permutations(range(dim)):
            fp = _permute_vector(flow_pred, perm)
            fg = _permute_vector(flow_gt, perm)
            dd = _permute_scalar(dist, perm)
            got = [float(x) for x in crit(fp, dd, fg, dd, mode="all")]
            for name, a, b in zip("AEB", base, got):
                assert abs(a - b) <= 1e-5 + 1e-4 * abs(a), (
                    f"loss{name} not permutation-invariant under {perm} (dim={dim}): {a} vs {b}")


def test_affinity_loss_axis_reflection_invariant():
    for dim in (2, 3):
        device = torch.device("cpu")
        crit = AffinityLoss(device, dim)
        flow_pred, flow_gt, dist = _asymmetric_pred_gt(dim, device)
        base = [float(x) for x in crit(flow_pred, dist, flow_gt, dist, mode="all")]
        for ax in range(dim):
            def reflect_v(F):
                Fr = torch.flip(F, dims=[2 + ax]).clone()
                Fr[:, ax] *= -1          # flip axis + negate that component
                return Fr
            fp = reflect_v(flow_pred)
            fg = reflect_v(flow_gt)
            dd = torch.flip(dist, dims=[1 + ax])
            got = [float(x) for x in crit(fp, dd, fg, dd, mode="all")]
            for name, a, b in zip("AEB", base, got):
                assert abs(a - b) <= 1e-5 + 1e-4 * abs(a), (
                    f"loss{name} not reflection-invariant on axis {ax} (dim={dim}): {a} vs {b}")


def _make_flow_dist(dim, device):
    """A simple radial-ish flow toward the centre + a distance-like field."""
    shape = (12,) * dim
    coords = torch.meshgrid(*[torch.arange(s, dtype=torch.float32) for s in shape],
                            indexing="ij")
    center = [(s - 1) / 2 for s in shape]
    disp = torch.stack([center[k] - coords[k] for k in range(dim)], 0)[None]  # (1,dim,*sp)
    mag = disp.pow(2).sum(1, keepdim=True).clamp_min(1e-6).sqrt()
    flow = 5.0 * disp / mag
    dist = (5 - mag[0]).clamp_min(0.0)  # (1,*sp)
    return flow.to(device), dist.to(device)


def test_affinity_loss_zero_for_identical_and_positive_for_perturbed():
    for dim in (2, 3):
        device = torch.device("cpu")
        crit = AffinityLoss(device, dim)
        flow, dist = _make_flow_dist(dim, device)

        a0, e0, b0 = crit(flow, dist, flow.clone(), dist.clone(), mode="all")
        assert float(a0) < 1e-6 and float(e0) < 1e-6, f"identical pred should be ~0 (dim={dim})"

        flow_bad = flow.clone()
        flow_bad[:, 0] = 0.0  # zero one component -> must register in lossE
        a1, e1, b1 = crit(flow_bad, dist, flow, dist, mode="all")
        assert float(e1) > 1e-4, f"perturbed flow must raise lossE (dim={dim})"
