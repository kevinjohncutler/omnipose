from .imports import *

from .fields import step_factor


def _sample_flow_grid(dP_t, positions, sizes, dim):
    """Sample flow at scattered positions using grid_sample (bilinear).

    Normalizes positions to [-1, 1] on-the-fly for each grid_sample call,
    rather than normalizing all coordinates upfront like steps_batch.
    """
    norm = 2.0 * positions / (sizes.unsqueeze(1) - 1) - 1.0  # (D, N)
    norm = norm.clamp(-1.0, 1.0)  # clamp so zeros padding is equivalent to border (MPS doesn't support border)
    if dim == 2:
        grid = torch.stack([norm[1], norm[0]], dim=-1)  # (N, 2)
        grid = grid.unsqueeze(0).unsqueeze(0)  # (1, 1, N, 2)
        sampled = torch.nn.functional.grid_sample(dP_t, grid, mode='bilinear',
                                                  align_corners=True, padding_mode='zeros')
        return sampled[0, :, 0, :]  # (D, N)
    elif dim == 3:
        grid = torch.stack([norm[2], norm[1], norm[0]], dim=-1)  # (N, 3)
        grid = grid.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, N, 3)
        sampled = torch.nn.functional.grid_sample(dP_t, grid, mode='bilinear',
                                                  align_corners=True, padding_mode='zeros')
        return sampled[0, :, 0, 0, :]  # (D, N)
    else:
        raise ValueError(f"Unsupported dim={dim}")


def _follow_flows_sparse(flow, mask, niter, device, suppress=False, interp=True):
    """Foreground-only Euler integration. Torch in, torch out.

    Pre-normalizes the flow field to normalized-coordinate units once before the
    loop — inside the loop it's just grid_sample + clamp per step, no per-step
    rescaling.  Supports suppress=True (step-damped momentum, matching steps_batch).

    Parameters
    ----------
    flow: (D, *spatial) torch tensor, flow field (already scaled, e.g. /5 or div_rescale output)
    mask: (*spatial) torch tensor, binary foreground mask
    niter: int, number of Euler steps
    device: torch device
    suppress: bool, if True use damped momentum integration matching steps_batch
    interp: bool, if True prefer bilinear; suppress=True forces nearest to match steps_batch

    Returns
    -------
    p: (D, *spatial) torch tensor, final pixel positions
    inds: tuple of torch tensors, foreground pixel indices
    """
    dim = flow.shape[0]
    spatial = flow.shape[1:]

    if mask is None:
        mask = flow.abs().sum(dim=0) > 0

    inds = torch.nonzero(mask, as_tuple=True)
    if len(inds[0]) == 0:
        return torch.zeros((dim,) + spatial, dtype=torch.float32, device=device), inds

    # Pre-normalize flow to normalized-coordinate units (2/(size-1) per dim).
    # In normalized space: pt_new = pt_old + flow_norm_at(pt_old) — no rescaling inside loop.
    shape = [s - 1.0 for s in spatial]
    scale = torch.tensor([2.0 / s for s in shape], dtype=torch.float32, device=device)
    flow_norm = flow.unsqueeze(0) * scale.view(1, dim, *([1] * len(spatial)))  # (1, D, *spatial)

    # Initial positions in normalized space
    pt = torch.stack([2.0 * inds[d].float() / shape[d] - 1.0 for d in range(dim)])  # (D, N)

    # Match steps_batch: suppress forces nearest interpolation
    mode = 'bilinear' if (interp and not suppress) else 'nearest'

    if dim == 2:
        def _grid(q): return torch.stack([q[1], q[0]], dim=-1).view(1, 1, -1, 2)
    else:
        def _grid(q): return torch.stack([q[2], q[1], q[0]], dim=-1).view(1, 1, 1, -1, 3)

    F_gs = torch.nn.functional.grid_sample

    if suppress:
        # Pre-sample at starting position (matches steps_batch's dPt0 initialization)
        dPt0 = F_gs(flow_norm, _grid(pt), mode=mode, align_corners=True,
                    padding_mode='zeros').view(dim, -1)

    for t in range(niter):
        dPt = F_gs(flow_norm, _grid(pt), mode=mode, align_corners=True,
                   padding_mode='zeros').view(dim, -1)

        if suppress:
            dPt = (dPt + dPt0) / 2.0
            dPt0.copy_(dPt)
            dPt = dPt / step_factor(t)

        pt = torch.clamp(pt + dPt, -1.0, 1.0)

    # Denormalize back to pixel coordinates
    positions = torch.stack([(pt[d] + 1.0) * 0.5 * shape[d] for d in range(dim)])

    # Write back to full spatial array (background pixels keep their grid coords)
    coords_grid = [torch.arange(s, dtype=torch.float32, device=device) for s in spatial]
    mesh = torch.meshgrid(coords_grid, indexing='ij')
    p = torch.stack(mesh)  # (D, *spatial)
    for d in range(dim):
        p[d][inds] = positions[d]

    return p, inds


def follow_flows_batch(dP, niter, omni=True, suppress=False, interp=True):
    """Batched dense Euler integration. GPU in, GPU out.

    Tracks every pixel in the image simultaneously across the batch.
    dP should already be scaled (e.g. divided by 5).

    Args:
        dP:    (B, D, *spatial) float tensor on device
        niter: int — integration steps

    Returns:
        p: (B, D, *spatial) float tensor on device — final pixel positions
    """
    B, D, *spatial = dP.shape
    N = 1
    for s in spatial:
        N *= s

    coords = [torch.arange(s, dtype=torch.float32, device=dP.device) for s in spatial]
    mesh   = torch.stack(torch.meshgrid(coords, indexing='ij'))  # (D, *spatial)
    # (B, D, N) — same initial grid for every image in batch
    p_flat = mesh.reshape(D, N).unsqueeze(0).expand(B, -1, -1).contiguous()

    final_p, _ = steps_batch(p_flat, dP, niter,
                              omni=omni, suppress=suppress, interp=interp,
                              calc_trace=False)  # (B, D, N)
    return final_p.reshape(B, D, *spatial)


def steps_batch(p, dP, niter, omni=True, suppress=True, interp=True,
                calc_trace=False, calc_bd=False, verbose=False):
    """Euler integration of pixel locations p subject to flow dP for niter steps in N dimensions."""
    align_corners = True

    interp = interp and not suppress
    mode = 'bilinear' if interp else 'nearest'

    d = dP.shape[1]
    shape = dP.shape[2:]
    inds = list(range(d))[::-1]

    shape = np.array(shape)[inds] - 1.
    B, D, I = p.shape
    pt = p[:, inds].permute(0, 2, 1).view([B] + [1] * (D - 1) + [I, D]).float()

    flow = dP[:, inds]

    for k in range(d):
        if shape[k] == 0:
            # Degenerate dimension (size 1): single pixel, no movement possible.
            pt[..., k] = 0.0   # [-1,1] centre for grid_sample align_corners=True
            flow[:, k] = 0.0
        else:
            pt[..., k] = 2 * pt[..., k] / shape[k] - 1
            flow[:, k] = 2 * flow[:, k] / shape[k]

    if calc_trace:
        dims = [-1, niter] + [-1] * (pt.ndim - 1)
        trace = torch.clone(pt).detach().unsqueeze(1).expand(*dims)

    if omni and suppress:
        dPt0 = torch.nn.functional.grid_sample(flow, pt, mode=mode, align_corners=align_corners)

    for t in range(niter):
        if calc_trace and t > 0:
            trace[:, t].copy_(pt)

        dPt = torch.nn.functional.grid_sample(flow, pt, mode=mode,
                                              align_corners=align_corners)

        if omni and suppress:
            dPt = (dPt + dPt0) / 2.
            dPt0.copy_(dPt)
            dPt /= step_factor(t)

        for k in range(d):
            pt[..., k] = torch.clamp(pt[..., k] + dPt[:, k], -1., 1.)

    pt = (pt + 1) * 0.5
    for k in range(d):
        pt[..., k] *= shape[k]

    if calc_trace:
        trace = (trace + 1) * 0.5
        for k in range(d):
            trace[..., k] *= shape[k]

    if calc_trace:
        tr = trace[..., inds].transpose(-1, 1).contiguous()
    else:
        tr = None
    p = pt[..., inds].transpose(-1, 1).contiguous()

    empty_cache()
    return p, tr
