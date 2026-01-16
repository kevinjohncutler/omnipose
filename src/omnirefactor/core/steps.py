from .imports import *

from .fields import step_factor

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
