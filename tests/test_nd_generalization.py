"""Tests for ND generalization fixes: affinity threshold and simplex illumination."""
import numpy as np
import pytest
import torch


# --------------------------------------------------------------------------- #
# _get_affinity_torch: `>= S - 2` threshold scales with dimension
# --------------------------------------------------------------------------- #

class TestAffinityThreshold:
    """The interior connectivity restoration threshold must scale with dim."""

    @staticmethod
    def _make_affinity_inputs(dim, size=16):
        """Create a simple solid block that should be fully interior-connected."""
        from omnipose.core.affinity import _get_affinity_torch
        from omnipose.utils.neighbor import kernel_setup, get_supporting_inds

        shape = (size,) * dim
        # Flow field: uniform rightward flow inside a solid block (unbatched)
        flow = np.zeros((dim,) + shape, dtype=np.float32)
        flow[-1] = 1.0  # flow along last axis

        # Initial/final pixel positions (Euler integration endpoints)
        grid = np.stack(np.meshgrid(
            *[np.arange(s, dtype=np.float32) for s in shape], indexing='ij'))
        initial = grid  # (D, *shape) — _ensure_torch will add batch dim
        final = initial + flow * 0.5

        dist = np.ones(shape, dtype=np.float32)
        iscell = np.ones(shape, dtype=np.float32)

        k = kernel_setup(dim)
        supporting_inds = get_supporting_inds(k.steps)

        return _get_affinity_torch, dict(
            initial=initial, final=final, flow=flow, dist=dist,
            iscell=iscell, steps=k.steps, fact=k.fact, inds=k.inds,
            supporting_inds=supporting_inds, niter=0,
            device=torch.device('cpu'),
        )

    def test_2d_interior_connected(self):
        fn, kw = self._make_affinity_inputs(dim=2, size=12)
        conn = fn(**kw)
        S = conn.shape[0]
        # Interior pixels (away from boundary) should have high connectivity
        interior = conn[:, 0, 3:-3, 3:-3]
        # Count connections per pixel (exclude center)
        counts = interior.sum(0)
        # Interior of uniform flow block should be mostly connected
        assert counts.float().mean() >= S - 3

    def test_3d_interior_connected(self):
        fn, kw = self._make_affinity_inputs(dim=3, size=10)
        conn = fn(**kw)
        S = conn.shape[0]
        interior = conn[:, 0, 3:-3, 3:-3, 3:-3]
        counts = interior.sum(0)
        assert counts.float().mean() >= S - 5

    def test_threshold_uses_S_not_hardcoded(self):
        """Verify the threshold is S-2, not hardcoded 7."""
        import inspect
        from omnipose.core.affinity import _get_affinity_torch
        source = inspect.getsource(_get_affinity_torch)
        # Strip comments before checking — the comment mentions the old value
        code_lines = [l.split('#')[0] for l in source.splitlines()]
        code_only = '\n'.join(code_lines)
        assert '>= 7' not in code_only, "Threshold should be S - 2, not hardcoded 7"
        assert 'S - 2' in code_only


# --------------------------------------------------------------------------- #
# OpenSimplex illumination: ND support + variance correction
# --------------------------------------------------------------------------- #

class TestSimplexIllumination:
    """Simplex noise illumination should work for 2D-4D with matched variance."""

    @staticmethod
    def _generate_illum_field(dim, seed=42, size=64, fs=30.0):
        from opensimplex import OpenSimplex
        from omnipose.transforms.augment import _SIMPLEX_NOISE
        from ocdkit.array import rescale

        simplex = OpenSimplex(seed=seed)
        shape = (size,) * dim
        coords = [np.arange(0, s, dtype=np.float32) / fs for s in shape[::-1]]
        noise_fn = getattr(simplex, _SIMPLEX_NOISE[dim])
        field = rescale(noise_fn(*coords))
        if dim > 2:
            field = np.clip(0.5 + (field - 0.5) * np.sqrt(dim / 2), 0, 1)
        return field

    def test_2d_produces_valid_field(self):
        field = self._generate_illum_field(2)
        assert field.ndim == 2
        assert field.min() >= 0 and field.max() <= 1

    def test_3d_produces_valid_field(self):
        field = self._generate_illum_field(3)
        assert field.ndim == 3
        assert field.min() >= 0 and field.max() <= 1

    def test_4d_produces_valid_field(self):
        field = self._generate_illum_field(4, size=16)
        assert field.ndim == 4
        assert field.min() >= 0 and field.max() <= 1

    def test_3d_std_matches_2d(self):
        """After sqrt(D/2) correction, 3D std should be within 15% of 2D."""
        stds_2d, stds_3d = [], []
        for seed in range(100):
            stds_2d.append(self._generate_illum_field(2, seed=seed).std())
            stds_3d.append(self._generate_illum_field(3, seed=seed).std())
        ratio = np.mean(stds_3d) / np.mean(stds_2d)
        assert 0.85 < ratio < 1.15, f"3D/2D std ratio {ratio:.3f} outside tolerance"

    def test_simplex_noise_dispatch(self):
        """_SIMPLEX_NOISE maps dim to correct method name."""
        from omnipose.transforms.augment import _SIMPLEX_NOISE
        assert _SIMPLEX_NOISE[2] == 'noise2array'
        assert _SIMPLEX_NOISE[3] == 'noise3array'
        assert _SIMPLEX_NOISE[4] == 'noise4array'
        assert 5 not in _SIMPLEX_NOISE  # falls back to linear gradient

    def test_fallback_to_linear_for_5d(self):
        """dim > 4 should not use simplex (not supported)."""
        from omnipose.transforms.augment import _SIMPLEX_NOISE
        assert 5 not in _SIMPLEX_NOISE
