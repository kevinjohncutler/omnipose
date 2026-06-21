"""Dimension-generic 3D payload builders for the ocdkit viewer.

Turns a volume + label volume (+ optional lineage links) into the JSON/binary
payloads the viewer renders: flow field, distance, spatial affinity graph,
cell-sink points, and temporal trajectories (lineage). Everything here is a
pure function operating on numpy arrays so it can be unit-tested headlessly
against the spacetime test stack — no model, no server, no GPU required (GPU is
optional and only accelerates the flow solve).

Shape conventions (spacetime: the leading axis is time "t"; for confocal it is
"z" — same math either way):
  - volume / masks / dist : ``(D, H, W)``  (D = depth/time)
  - flow field ``mu``      : ``(3, D, H, W)`` = ``[d_depth, dy, dx]``
  - affinity ``spatial``   : ``(S, D, H, W)`` uint8, S = 3**dim - 1 = 26 (center dropped)
  - steps                  : ``(S, 3)`` int8 neighbour offsets ``[dz, dy, dx]``
  - points                 : ``(N, 3)`` float32 ``[z, y, x]``

Binary payloads are gzip+base64 (decode in-browser with DecompressionStream or
pako). Per-voxel volumes (image, mask, affinity) can be large; ``build_bundle``
embeds them only when ``embed_volumes`` / ``embed_affinity`` is set (fine for
crops + tests). Production serves intensity/mask/affinity per-slice from a
held-in-memory volume via the route layer — see ``affinity_volume`` /
``slice_rgb_png`` which the routes call on demand.
"""
from __future__ import annotations

import base64
import gzip
import io
from typing import Any, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# encoding helpers
# ---------------------------------------------------------------------------

def encode_array(arr: np.ndarray, *, gzip_it: bool = True, level: int = 1) -> dict[str, Any]:
    """Pack an ndarray as ``{dtype, shape, gzip, b64}`` (C-order bytes)."""
    arr = np.ascontiguousarray(arr)
    raw = arr.tobytes()
    if gzip_it:
        raw = gzip.compress(raw, level)
    return {
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "gzip": bool(gzip_it),
        "b64": base64.b64encode(raw).decode("ascii"),
    }


def decode_array(d: Mapping[str, Any]) -> np.ndarray:  # type: ignore[name-defined]
    """Inverse of :func:`encode_array` (used by tests and any Python consumer)."""
    raw = base64.b64decode(d["b64"])
    if d.get("gzip"):
        raw = gzip.decompress(raw)
    return np.frombuffer(raw, dtype=np.dtype(d["dtype"])).reshape(tuple(d["shape"]))


def _narrow_label_dtype(masks: np.ndarray) -> np.ndarray:
    """Down-cast a label volume to the smallest uint that holds ``max_label``."""
    m = np.asarray(masks)
    mx = int(m.max()) if m.size else 0
    if mx <= np.iinfo(np.uint8).max:
        dt = np.uint8
    elif mx <= np.iinfo(np.uint16).max:
        dt = np.uint16
    else:
        dt = np.uint32
    return m.astype(dt, copy=False)


def _to_numpy(x: Any) -> np.ndarray:
    """Detach a torch tensor or pass through a numpy array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


# ---------------------------------------------------------------------------
# kernel / steps
# ---------------------------------------------------------------------------

def kernel_steps(dim: int) -> tuple[np.ndarray, int]:
    """Non-centre neighbour offsets for ``dim`` and the centre index.

    Returns ``(steps, center_idx)`` where ``steps`` is ``(3**dim - 1, dim)``
    int8 with the centre (zero) offset removed. dim=2 -> 8 steps, dim=3 -> 26.
    """
    from omnipose import utils

    steps, inds, idx, fact, sign = utils.kernel_setup(dim)
    steps = np.asarray(steps)
    center = int(idx)
    keep = np.ones(steps.shape[0], dtype=bool)
    keep[center] = False
    return np.ascontiguousarray(steps[keep].astype(np.int8)), center


# ---------------------------------------------------------------------------
# flow + distance (derived from a label volume via the eikonal solve)
# ---------------------------------------------------------------------------

def flow_and_dist(masks: np.ndarray, *, use_gpu: bool = False,
                  device: Any = None) -> tuple[np.ndarray, np.ndarray]:
    """Compute the flow field and distance from a label volume.

    Returns ``(mu, dist)`` with ``mu`` shape ``(dim, *spatial)`` float32
    (``[d_depth, dy, dx]`` for 3D) and ``dist`` shape ``(*spatial,)`` float32.
    """
    from omnipose import core

    m = np.ascontiguousarray(np.asarray(masks).astype(np.int32))
    dim = m.ndim
    res = core.masks_to_flows(m, dim=dim, use_gpu=use_gpu, device=device)
    mu = _to_numpy(res.mu).astype(np.float32, copy=False)
    dist = _to_numpy(res.dists).astype(np.float32, copy=False)
    return mu, dist


# ---------------------------------------------------------------------------
# spatial affinity graph (S, *spatial)
# ---------------------------------------------------------------------------

def affinity_volume(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Spatial affinity graph for a label volume.

    Returns ``(steps, spatial)`` where ``steps`` is ``(S, dim)`` int8 and
    ``spatial`` is ``(S, *spatial)`` uint8 (1 = neighbour shares the label),
    centre offset removed (S = 3**dim - 1).

    NOTE: ``(S, *spatial)`` is large for full volumes (26 x voxels). Callers
    that need the whole graph (tests, small crops) use this directly; the live
    viewer computes it once and serves in-plane slices on demand.
    """
    from omnipose import core, utils

    m = np.ascontiguousarray(np.asarray(masks).astype(np.int32))
    dim = m.ndim
    steps, inds, idx, fact, sign = utils.kernel_setup(dim)
    coords = np.nonzero(m)
    if coords[0].size == 0:
        steps = np.asarray(steps)
        keep = np.ones(steps.shape[0], dtype=bool)
        keep[int(idx)] = False
        S = int(keep.sum())
        return (np.ascontiguousarray(steps[keep].astype(np.int8)),
                np.zeros((S,) + m.shape, dtype=np.uint8))
    aff = core.masks_to_affinity(m, coords, steps, inds, idx, fact, sign, dim)
    spatial = core.spatial_affinity(aff, coords, m.shape)
    steps = np.asarray(steps)
    center = int(idx)
    keep = np.ones(steps.shape[0], dtype=bool)
    keep[center] = False
    steps_nc = np.ascontiguousarray(steps[keep].astype(np.int8))
    spatial_nc = np.ascontiguousarray((spatial[keep] > 0).astype(np.uint8))
    return steps_nc, spatial_nc


# ---------------------------------------------------------------------------
# flow -> RGB
# ---------------------------------------------------------------------------

def _rgb_flow_2d(dp2: np.ndarray) -> np.ndarray:
    """Sinebow colouring of an in-plane 2-vector field ``(2, H, W)`` -> ``(H,W,3)``.

    Mirrors ocdkit ``plot.color.rgb_flow`` (complex-plane -> 3 cosine bases) but
    is dependency-light and numpy-only so it works without torch in tests.
    """
    dy, dx = dp2[0], dp2[1]
    mag = np.sqrt(dy * dy + dx * dx)
    ang = np.arctan2(dy, dx)  # [-pi, pi]
    roots = (2.0 * np.arange(3) / 3.0 + 1.0) * np.pi
    rgb = np.stack([(np.cos(ang - r) * 0.5 + 0.5) for r in roots], axis=-1)
    mx = float(np.percentile(mag, 99)) if mag.size else 1.0
    scale = np.clip(mag / (mx + 1e-9), 0.0, 1.0)
    rgb = rgb * scale[..., None]
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def flow_rgb_slices(mu: np.ndarray) -> np.ndarray:
    """Per-depth in-plane (dy,dx) flow RGB for the 2.5D slice view.

    ``mu`` ``(3, D, H, W)`` -> ``(D, H, W, 3)`` uint8. Uses only the in-plane
    components so each slice looks exactly like the familiar 2D flow overlay.
    """
    mu = np.asarray(mu)
    if mu.ndim == 3:  # already 2D (2,H,W) -> single slice
        return _rgb_flow_2d(mu[:2])[None]
    D = mu.shape[1]
    out = np.empty((D, mu.shape[2], mu.shape[3], 3), dtype=np.uint8)
    inplane = mu[1:3]  # (2, D, H, W) = (dy, dx)
    for z in range(D):
        out[z] = _rgb_flow_2d(inplane[:, z])
    return out


def rgb_flow_3d(mu: np.ndarray) -> np.ndarray:
    """Directional colour for the full 3-vector field (for the volume view).

    ``mu`` ``(3, D, H, W)`` -> ``(D, H, W, 3)`` uint8. Maps the unit direction to
    RGB (axis -> channel) and modulates brightness by normalised magnitude, so
    a 3-vector gets a stable colour (no complex-plane degeneracy).
    """
    v = np.asarray(mu, dtype=np.float32)
    mag = np.sqrt((v * v).sum(axis=0))
    unit = v / (mag + 1e-9)
    rgb = unit * 0.5 + 0.5  # (3, D, H, W) in [0,1], axis->channel
    mx = float(np.percentile(mag, 99)) if mag.size else 1.0
    scale = np.clip(mag / (mx + 1e-9), 0.0, 1.0)
    rgb = rgb * scale[None]
    return (np.clip(np.moveaxis(rgb, 0, -1), 0, 1) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# distance -> magma PNG slices
# ---------------------------------------------------------------------------

def _magma_lut() -> np.ndarray:
    from matplotlib import pyplot as plt
    cmap = plt.get_cmap("magma")
    return cmap(np.linspace(0, 1, cmap.N))[:, :4]


def dist_rgb_slices(dist: np.ndarray) -> np.ndarray:
    """Per-depth magma-coloured distance ``(D,H,W)`` -> ``(D,H,W,3)`` uint8.

    Normalised over the whole volume so brightness is comparable across slices.
    """
    d = np.asarray(dist, dtype=np.float32)
    if d.ndim == 2:
        d = d[None]
    finite = np.isfinite(d)
    lo = float(d[finite].min()) if finite.any() else 0.0
    hi = float(d[finite].max()) if finite.any() else 1.0
    norm = (d - lo) / (hi - lo) if hi > lo else np.zeros_like(d)
    norm = np.clip(norm, 0.0, 1.0)
    lut = _magma_lut()
    idx = np.round(norm * (len(lut) - 1)).astype(int)
    rgb = (lut[idx][..., :3] * 255).astype(np.uint8)
    return rgb


# ---------------------------------------------------------------------------
# points (cell sinks) from an Euler-integration field p
# ---------------------------------------------------------------------------

def reconstruct_points(masks: np.ndarray, mu: np.ndarray, dist: np.ndarray, *,
                       use_gpu: bool = False, device: Any = None,
                       niter: Optional[int] = None, mask_threshold: float = 0.0,
                       flow_threshold: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Run omnipose flow-following on a derived flow field to get REAL sinks.

    Returns ``(recon_mask, p)`` where ``p`` ``(dim, *spatial)`` is the converged
    (flow-followed) position of every voxel — the real trajectory endpoints,
    as opposed to GT centroids. Uses ``affinity_seg=False`` (the 3D
    affinity-reconstruction path has a divergence-shape bug; the flow-following
    path is unaffected).
    """
    from omnipose import core

    m = np.ascontiguousarray(np.asarray(masks).astype(np.int32))
    dim = m.ndim
    out = core.compute_masks(
        dP=np.ascontiguousarray(np.asarray(mu, np.float32)),
        dist=np.ascontiguousarray(np.asarray(dist, np.float32)),
        bd=None, niter=niter, mask_threshold=mask_threshold,
        flow_threshold=flow_threshold, affinity_seg=False, cluster=False,
        omni=True, dim=dim, nclasses=2, use_gpu=use_gpu, device=device,
        calc_trace=False, verbose=False)
    recon_mask = np.asarray(out[0]).astype(np.int32)
    p = _to_numpy(out[1]).astype(np.float32)
    return recon_mask, p


def points_from_p(masks: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Foreground cell-sink coordinates as ``(N, 3)`` float32 ``[z, y, x]``.

    ``p`` is the flow-followed position field ``(dim, *spatial)``. Returns the
    converged position of every foreground voxel.
    """
    m = np.asarray(masks)
    p = np.asarray(p, dtype=np.float32)
    dim = m.ndim
    fg = np.nonzero(m > 0)
    if fg[0].size == 0:
        return np.zeros((0, dim), dtype=np.float32)
    comps = [np.clip(p[c][fg], 0, m.shape[c] - 1) for c in range(dim)]
    return np.ascontiguousarray(np.stack(comps, axis=-1).astype(np.float32))


# ---------------------------------------------------------------------------
# trajectories / lineage
# ---------------------------------------------------------------------------

def parse_links(links_path: str) -> list[list[int]]:
    """Parse a ``parent,daughter`` edge list (one per line)."""
    edges: list[list[int]] = []
    with open(links_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(" ", "").split(",")
            if len(parts) != 2:
                continue
            try:
                edges.append([int(parts[0]), int(parts[1])])
            except ValueError:
                continue
    return edges


def trajectories(masks: np.ndarray,
                 edges: Optional[Sequence[Sequence[int]]] = None,
                 links_path: Optional[str] = None) -> dict[str, Any]:
    """Per-label centroid tracks through the depth/time axis + lineage edges.

    ``masks`` ``(D, H, W)``. Returns
    ``{tracks: [{label, frames:[d...], centroids:[[y,x]...]}], edges:[[p,d]...]}``.
    A track's centroids[i] is the (y,x) centre of mass of that label in frame
    frames[i]; the frame axis is depth/time.
    """
    from scipy import ndimage as ndi

    m = np.asarray(masks)
    if m.ndim != 3:
        raise ValueError("trajectories expects a (D,H,W) label volume")
    D = m.shape[0]
    tracks: dict[int, dict[str, list]] = {}
    for d in range(D):
        plane = m[d]
        labs = np.unique(plane)
        labs = labs[labs > 0]
        if labs.size == 0:
            continue
        coms = ndi.center_of_mass(np.ones_like(plane), labels=plane,
                                  index=labs.tolist())
        for lab, com in zip(labs.tolist(), np.atleast_2d(coms)):
            t = tracks.setdefault(int(lab), {"frames": [], "centroids": []})
            t["frames"].append(int(d))
            t["centroids"].append([float(com[0]), float(com[1])])  # (y, x)
    track_list = [
        {"label": lab, "frames": t["frames"], "centroids": t["centroids"]}
        for lab, t in sorted(tracks.items())
    ]
    if edges is None and links_path is not None:
        edges = parse_links(links_path)
    edges_out = [[int(a), int(b)] for a, b in (edges or [])]
    return {"tracks": track_list, "edges": edges_out}


# ---------------------------------------------------------------------------
# bundle
# ---------------------------------------------------------------------------

def build_bundle(volume: Optional[np.ndarray],
                 masks: np.ndarray,
                 *,
                 links_path: Optional[str] = None,
                 edges: Optional[Sequence[Sequence[int]]] = None,
                 use_gpu: bool = False,
                 device: Any = None,
                 do_flow: bool = True,
                 do_affinity: bool = True,
                 do_trajectories: bool = True,
                 do_recon: bool = False,
                 max_points: int = 8000,
                 embed_volumes: bool = True,
                 embed_affinity: bool = True) -> dict[str, Any]:
    """Assemble the full 3D viewer bundle from a volume + label volume.

    The flags let the route layer skip / defer the heavy parts (e.g. serve
    affinity and intensity per-slice instead of embedding them whole).
    """
    m = np.asarray(masks)
    dim = m.ndim
    D, H, W = (m.shape if dim == 3 else (1, *m.shape))
    steps, _ = kernel_steps(dim)

    bundle: dict[str, Any] = {
        "meta": {
            "dim": int(dim),
            "axes": (["t", "y", "x"] if dim == 3 else ["y", "x"]),
            "depth": int(D), "height": int(H), "width": int(W),
            "nLabels": int(m.max()) if m.size else 0,
        },
        "steps": steps.tolist(),
    }

    mask_nc = _narrow_label_dtype(m)
    bundle["mask"] = encode_array(mask_nc) if embed_volumes else {
        "deferred": True, "dtype": str(mask_nc.dtype),
        "shape": list(mask_nc.shape)}

    if volume is not None:
        vol = np.asarray(volume)
        bundle["image"] = encode_array(vol) if embed_volumes else {
            "deferred": True, "dtype": str(vol.dtype), "shape": list(vol.shape)}

    if do_flow:
        mu, dist = flow_and_dist(m, use_gpu=use_gpu, device=device)
        bundle["flow"] = {
            "rgbSlices": encode_array(flow_rgb_slices(mu)),
            "rgb3d": encode_array(rgb_flow_3d(mu)),
            "raw": encode_array(mu.astype(np.float16)),
        }
        bundle["distance"] = {"rgbSlices": encode_array(dist_rgb_slices(dist))}

    if do_affinity:
        astep, aspatial = affinity_volume(m)
        bundle["affinity"] = {
            "steps": astep.tolist(),
            "stepCount": int(astep.shape[0]),
            "spatial": (encode_array(aspatial) if embed_affinity else {
                "deferred": True, "shape": list(aspatial.shape)}),
        }

    if do_trajectories and dim == 3:
        bundle["trajectories"] = trajectories(m, edges=edges, links_path=links_path)

    if do_recon:
        # Real omnipose flow-following: where each voxel converges (cell sinks),
        # vs GT centroids. Populates the points overlay with real recon data.
        if not do_flow:
            mu, dist = flow_and_dist(m, use_gpu=use_gpu, device=device)
        recon_mask, p = reconstruct_points(m, mu, dist, use_gpu=use_gpu, device=device)
        coords = points_from_p(recon_mask, p)            # (N, dim) [z,y,x]
        n = coords.shape[0]
        if n > max_points:
            idx = np.linspace(0, n - 1, max_points).astype(np.int64)
            coords = coords[idx]
        bundle["points"] = {**encode_array(coords.astype(np.float32)),
                            "count": int(coords.shape[0]), "total": int(n)}

    return bundle


def bundle_from_files(raw_path: Optional[str], masks_path: str,
                      links_path: Optional[str] = None, **kwargs: Any) -> dict[str, Any]:
    """Convenience entry point for the route layer: read tiffs, build a bundle.

    Auto-detects a sibling ``*_links.txt`` next to ``masks_path`` when
    ``links_path`` is not given.
    """
    import os
    import tifffile

    masks = tifffile.imread(masks_path)
    volume = tifffile.imread(raw_path) if raw_path else None
    if links_path is None:
        guess = masks_path
        for suffix in ("_masks.tif", "_masks.tiff", ".tif", ".tiff"):
            if guess.endswith(suffix):
                guess = guess[: -len(suffix)]
                break
        cand = guess + "_links.txt"
        if os.path.exists(cand):
            links_path = cand
    return build_bundle(volume, masks, links_path=links_path, **kwargs)
