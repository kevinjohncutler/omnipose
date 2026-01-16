from .imports import *

# @njit('(float64[:], int32[:], int32[:], int32, int32, int32, int32)', nogil=True)
def _extend_centers(T, y, x, ymed, xmed, Lx, niter):
    """Run diffusion from center of mask (ymed, xmed) on mask pixels (y, x)."""
    for t in range(niter):
        T[ymed * Lx + xmed] += 1
        T[y * Lx + x] = 1 / 9. * (T[y * Lx + x] + T[(y - 1) * Lx + x] + T[(y + 1) * Lx + x] +
                                  T[y * Lx + x - 1] + T[y * Lx + x + 1] +
                                  T[(y - 1) * Lx + x - 1] + T[(y - 1) * Lx + x + 1] +
                                  T[(y + 1) * Lx + x - 1] + T[(y + 1) * Lx + x + 1])
    return T



def _extend_centers_gpu(neighbors, centers, isneighbor, Ly, Lx, n_iter=200, device=torch.device('cuda')):
    """Runs diffusion on GPU to generate flows for training images or quality control."""
    if device is not None:
        device = device
    nimg = neighbors.shape[0] // 9
    pt = torch.from_numpy(neighbors).to(device)

    T = torch.zeros((nimg, Ly, Lx), dtype=torch.double, device=device)
    meds = torch.from_numpy(centers.astype(int)).to(device).long()
    isneigh = torch.from_numpy(isneighbor).to(device)
    for i in range(n_iter):
        T[:, meds[:, 0], meds[:, 1]] += 1
        Tneigh = T[:, pt[:, :, 0], pt[:, :, 1]]
        Tneigh *= isneigh
        T[:, pt[0, :, 0], pt[0, :, 1]] = Tneigh.mean(axis=1)

    del meds, isneigh, Tneigh
    T = torch.log(1. + T)
    # gradient positions
    grads = T[:, pt[[2, 1, 4, 3], :, 0], pt[[2, 1, 4, 3], :, 1]]
    dy = grads[:, 0] - grads[:, 1]
    dx = grads[:, 2] - grads[:, 3]

    del grads
    mu_torch = np.stack((dy.cpu().squeeze(), dx.cpu().squeeze()), axis=-2)
    return mu_torch
