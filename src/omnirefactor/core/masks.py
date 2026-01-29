import time

import numpy as np
import torch
import fastremap
import ncolor
from scipy.ndimage import binary_dilation, binary_fill_holes, find_objects, label, maximum_filter1d, mean, zoom
from skimage import filters, measure
from skimage.morphology import remove_small_holes
from skimage.segmentation import find_boundaries
from sklearn.neighbors import NearestNeighbors

from dbscan import DBSCAN as new_DBSCAN

from .. import utils
from ..gpu import empty_cache
from ..logger import get_logger
from .affinity import (
    _despur,
    _get_affinity_torch,
    affinity_to_boundary,
    affinity_to_masks,
    boundary_to_masks,
)
from .diam import diameters, dist_to_diam
from .fields import div_rescale, step_factor
from .flows import masks_to_flows
from .steps import steps_batch

try:
    from hdbscan import HDBSCAN
    HDBSCAN_ENABLED = True
except ModuleNotFoundError:
    HDBSCAN_ENABLED = False

SKLEARN_ENABLED = True
SKIMAGE_ENABLED = True

omnipose_logger = get_logger('core')


def compute_masks(dP, dist, affinity_graph=None, bd=None, p=None, coords=None, iscell=None, niter=None, rescale_factor=1.0, resize=None,
                  mask_threshold=0.0, diam_threshold=12., flow_threshold=0.4,
                  interp=True, cluster=False, affinity_seg=False, do_3D=False,
                  min_size=None, max_size=None, hole_size=None, omni=True,
                  calc_trace=False, verbose=False, use_gpu=False, device=None, nclasses=2,
                  dim=2, eps=None, hdbscan=False, flow_factor=6, debug=False, override=False, suppress=None, despur=False):
    """
    Compute masks using dynamics from dP, dist, and boundary outputs.
    Called in cellpose.models().
    """

    pad = 0
    if do_3D:
        dim = 3
    pad_seq = [(0,) * 2] + [(pad,) * 2] * dim
    unpad = tuple([slice(pad, -pad) if pad else slice(None, None)] * dim)

    if hole_size is None:
        hole_size = 3 ** (dim // 2)

    labels = None

    if verbose:
        startTime0 = time.time()
        omnipose_logger.info(f'mask_threshold is {mask_threshold}')
        if omni and (not SKIMAGE_ENABLED):
            omnipose_logger.warning('Omni enabled but skimage not enabled')

    if iscell is None:
        if coords is not None:
            iscell = np.zeros_like(dist, dtype=np.int32)
            iscell[tuple(coords)] = 1
        else:
            if (omni and SKIMAGE_ENABLED) or override:
                if verbose:
                    omnipose_logger.info('Using hysteresis threshold.')
                iscell = filters.apply_hysteresis_threshold(dist, mask_threshold - 1, mask_threshold)

            else:
                iscell = dist > mask_threshold

    if np.any(iscell) and nclasses > 1:

        iscell_pad = np.pad(iscell, pad)
        coords = np.array(np.nonzero(iscell_pad)).astype(np.int32)
        shape = iscell_pad.shape

        if suppress is None:
            suppress = omni and not affinity_seg

        if omni:
            if suppress:
                dP_ = div_rescale(dP, iscell) / rescale_factor
            else:
                dP_ = dP.copy() / 5.

            if dim > 2 and suppress:
                dP_ *= flow_factor
                print('dP_ times {} for >2d, still experimenting'.format(flow_factor))

        else:
            dP_ = dP * iscell / 5.

        dP_pad = np.pad(dP_, pad_seq)
        dt_pad = np.pad(dist, pad)
        bd_pad = np.pad(bd, pad)
        bounds = None

        if (cluster or affinity_seg or not suppress) and niter is None:
            niter = int(diameters(iscell, dist) / (1 + affinity_seg))

        if p is None:
            p, coords, tr = follow_flows(dP_pad, dt_pad, coords, niter=niter, interp=interp,
                                         use_gpu=use_gpu, device=device, omni=omni,
                                         suppress=suppress,
                                         calc_trace=calc_trace, verbose=verbose)
        else:
            tr = []
            if verbose:
                omnipose_logger.info('p given')

            p[:, ~iscell_pad] = np.stack(np.nonzero(~iscell_pad))

        if omni or override:
            steps, inds, idx, fact, sign = utils.kernel_setup(dim)
            if affinity_seg:
                hole_size = 0
                if affinity_graph is None:
                    if verbose:
                        omnipose_logger.info('computing affinity graph')

                    initial_points = np.stack(_meshgrid(iscell_pad.shape))
                    final_points = p
                    supporting_inds = utils.get_supporting_inds(steps)

                    affinity_graph = _get_affinity_torch(initial_points,
                                                         final_points,
                                                         dP_pad,
                                                         dt_pad,
                                                         iscell_pad,
                                                         steps,
                                                         fact,
                                                         inds,
                                                         supporting_inds,
                                                         niter,
                                                         device=device,
                                                         )
                    affinity_graph = affinity_graph.squeeze().cpu().numpy()
                    affinity_graph = affinity_graph[(Ellipsis,) + tuple(coords)]

                neighbors = utils.get_neighbors(tuple(coords), steps, dim, shape, pad=pad)
                indexes, neigh_inds, ind_matrix = utils.get_neigh_inds(tuple(neighbors), tuple(coords), shape)

                despur = dim == 2 and despur
                if verbose and not despur:
                    omnipose_logger.info('despur disabled')

                if despur:
                    non_self = np.array(list(set(np.arange(len(steps))) - {inds[0][0]}))
                    cardinal = np.concatenate(inds[1:2])
                    ordinal = np.concatenate(inds[2:])

                    affinity_graph = _despur(affinity_graph,
                                             neigh_inds,
                                             indexes,
                                             steps,
                                             non_self,
                                             cardinal,
                                             ordinal,
                                             dim)

                bounds = affinity_to_boundary(iscell_pad, affinity_graph, tuple(coords))

                if cluster:
                    labels = affinity_to_masks(affinity_graph, neigh_inds, iscell_pad, coords, verbose=verbose)
                else:
                    if verbose:
                        omnipose_logger.info('doing affinity seg without cluster.')
                    labels, bounds, _ = boundary_to_masks(bounds, iscell_pad)

            else:
                labels, _ = get_masks(p, bd_pad, dt_pad, iscell_pad, coords, nclasses, cluster=cluster,
                                      diam_threshold=diam_threshold, verbose=verbose,
                                      eps=eps, hdbscan=hdbscan)
                affinity_graph = None
                coords = np.nonzero(labels)
        else:
            # should deprecate this code path, remove omni toggle, maybe have a cp toggle instead or combine suppress=False with CC
            labels = get_masks_cp(p, iscell=iscell_pad,
                                  flows=dP_pad if flow_threshold > 0 else None,
                                  use_gpu=use_gpu)

        if not do_3D:
            flows = np.pad(dP, pad_seq)
            shape0 = flows.shape[1:]
            if labels.max() > 0 and flow_threshold is not None and flow_threshold > 0 and flows is not None:
                labels = remove_bad_flow_masks(labels, flows,
                                               coords=coords,
                                               affinity_graph=affinity_graph,
                                               threshold=flow_threshold,
                                               use_gpu=use_gpu,
                                               device=device,
                                               omni=omni)
                _, labels = np.unique(labels, return_inverse=True)
                labels = np.reshape(labels, shape0).astype(np.int32)

        masks = utils.fill_holes_and_remove_small_masks(labels, min_size=min_size, max_size=max_size,
                                                  hole_size=hole_size, dim=dim) * iscell_pad
        resize_pad = np.array([r + 2 * pad for r in resize]) if resize is not None else labels.shape
        if tuple(resize_pad) != labels.shape:
            if verbose:
                omnipose_logger.info(f'resizing output with resize = {resize_pad}')
            ratio = np.array(resize_pad) / np.array(labels.shape)
            masks = zoom(masks, ratio, order=0).astype(np.int32)
            iscell_pad = masks > 0
            dt_pad = zoom(dt_pad, ratio, order=1)
            dP_pad = zoom(dP_pad, np.concatenate([[1], ratio]), order=1)

            if verbose and affinity_seg:
                omnipose_logger.info('affinity_seg not compatible with rescaling, disabling')
            affinity_seg = False

        if not affinity_seg:
            bounds = find_boundaries(masks, mode='inner', connectivity=dim)
            # todo: replace this with masks to affinity followed by affinity to boundary? 
        
        # if bounds is None:
        #     if verbose:
        #         print('Default clustering on, finding boundaries via affinity.')
        #     print('TO-DO: replace with _get_affinity_torch')
        #     affinity_graph, neighbors, neigh_inds, bounds = _get_affinity(steps, masks, dP_pad, dt_pad, p, inds, pad=pad)

        #     gone = neigh_inds[3 ** dim // 2, np.sum(affinity_graph, axis=0) == 0]
        #     crd = coords.T
        #     masks[tuple(crd[gone].T)] = 0
        #     iscell_pad[tuple(crd[gone].T)] = 0
        # else:
        #     bounds *= masks > 0

        fastremap.renumber(masks, in_place=True)

        masks_unpad = masks[unpad] if pad else masks
        bounds_unpad = bounds[unpad] if pad else bounds

        if affinity_seg:
            coords_remaining = np.nonzero(masks)
            inds_remaining = ind_matrix[coords_remaining]
            affinity_graph_unpad = affinity_graph[:, inds_remaining]
            neighbors_unpad = neighbors[..., inds_remaining] - pad

            augmented_affinity = np.vstack((neighbors_unpad, affinity_graph_unpad[np.newaxis]))

            if calc_trace:
                print('warning calc trace not cropped')

        else:
            augmented_affinity = []

        ret = [masks_unpad, p, tr, bounds_unpad, augmented_affinity]

    else:
        omnipose_logger.info('No cell pixels found.')
        ret = [iscell, np.zeros([2, 1, 1]), [], iscell, []]

    if debug:
        ret += [labels]

    if verbose:
        executionTime0 = (time.time() - startTime0)
        omnipose_logger.info('compute_masks() execution time: {:.3g} sec'.format(executionTime0))
        if labels is not None:
            omnipose_logger.info('\texecution time per pixel: {:.6g} sec/px'.format(executionTime0 / np.prod(labels.shape)))
            omnipose_logger.info('\texecution time per cell pixel: {:.6g} sec/px'.format(np.nan if not np.count_nonzero(labels) else executionTime0 / np.count_nonzero(labels)))
        else:
            omnipose_logger.info('\tno objects found')

    return (*ret,)


def get_masks(p, bd, dist, mask, inds, nclasses=2, cluster=False,
              diam_threshold=12., eps=None, min_samples=5, hdbscan=False, verbose=False):
    """Omnipose mask recontruction algorithm."""
    if nclasses > 1:
        dt = np.abs(dist[mask])
        d = dist_to_diam(dt, mask.ndim)

    else:
        d = diameters(mask, dist)

    if eps is None:
        eps = 2 ** 0.5
    if verbose:
        omnipose_logger.info('Mean diameter is %f' % d)

    if d <= diam_threshold:
        cluster = True
        if verbose and not cluster:
            omnipose_logger.info('Turning on subpixel clustering for label continuity.')

    cell_px = tuple(inds)
    coords = np.nonzero(mask)
    newinds = p[(Ellipsis,) + cell_px].T
    mask = np.zeros(p.shape[1:], np.uint32)

    if verbose:
        omnipose_logger.info('cluster: {}, SKLEARN_ENABLED: {}'.format(cluster, SKLEARN_ENABLED))

    if cluster and SKLEARN_ENABLED:
        if verbose:
            startTime = time.time()
            alg = ['', 'H']
            omnipose_logger.info('Doing {}DBSCAN clustering with eps={}, min_samples={}'.format(alg[hdbscan], eps, min_samples))

        if hdbscan and not HDBSCAN_ENABLED:
            omnipose_logger.warning('HDBSCAN clustering requested but not installed. Defaulting to DBSCAN')

        if hdbscan and HDBSCAN_ENABLED:
            clusterer = HDBSCAN(cluster_selection_epsilon=eps,
                                min_samples=min_samples)

            clusterer.fit(newinds)
            labels = clusterer.labels_
        else:
            labels, _ = new_DBSCAN(newinds, eps=eps, min_samples=min_samples)

        if verbose:
            executionTime = (time.time() - startTime)
            omnipose_logger.info('Execution time in seconds: ' + str(executionTime))
            omnipose_logger.info('{} unique labels found'.format(len(np.unique(labels)) - 1))

        snap = 1
        if snap:
            nearest_neighbors = NearestNeighbors(n_neighbors=5)
            neighbors = nearest_neighbors.fit(newinds)
            o_inds = np.where(labels == -1)[0]
            if len(o_inds):
                outliers = [newinds[i] for i in o_inds]
                nearest_dists, nearest_indices = neighbors.kneighbors(outliers)

                nearest_labels = labels[nearest_indices]

                nearest_idx = [np.where(n != -1)[0][0] if np.any(n != -1) else 0 for n in nearest_labels]
                dist_thresh = eps
                l = [nl[i] if nd[i] < dist_thresh else -1 for i, nl, nd in zip(nearest_idx, nearest_labels, nearest_dists)]
                labels[o_inds] = l
                if verbose:
                    omnipose_logger.info(f'Outlier cleanup with dist threshold {dist_thresh:.2f}:')
                    distances = [nd[i] for i, nd in zip(nearest_idx, nearest_dists)]
                    omnipose_logger.info(f'\tmin and max distance to nearest cluster: {np.min(distances):.2f},{np.max(distances):.2f}')
                    omnipose_logger.info('\tSnapped {} of {} outliers to nearest cluster'.format(np.sum(np.array(l) != -1), len(o_inds)))

        mask[cell_px] = labels + 1

    else:
        newinds = np.rint(newinds.T).astype(int)
        new_px = tuple(newinds)
        skelmask = np.zeros_like(dist, dtype=bool)
        skelmask[new_px] = 1

        border_mask = np.zeros(skelmask.shape, dtype=bool)
        border_px = border_mask.copy()
        border_mask = binary_dilation(border_mask, border_value=1, iterations=5)

        border_px[border_mask] = skelmask[border_mask]
        if verbose:
            omnipose_logger.info('nclasses: {}, mask.ndim: {}'.format(nclasses, mask.ndim))
        if nclasses == mask.ndim + 2:
            border_px[bd > -1] = 0

        skelmask[border_mask] = border_px[border_mask]

        if SKIMAGE_ENABLED:
            cnct = skelmask.ndim
            labels = measure.label(skelmask, connectivity=cnct)
        else:
            labels = label(skelmask)[0]
        mask[cell_px] = labels[new_px]

    if verbose:
        omnipose_logger.info('Done finding masks.')
    return mask, labels


# Generalizing to ND. Again, torch required but should be plenty fast on CPU too compared to jitted but non-explicitly-parallelized CPU code.
# grid_sample will only work for up to 5D tensors (3D segmentation). Will have to address this shortcoming if we ever do 4D.



def follow_flows(dP, dist, inds, niter=None, interp=True, use_gpu=True,
                 device=None, omni=True, suppress=False, calc_trace=False, verbose=False):
    """Define pixels and run dynamics to recover masks in 2D."""
    if verbose:
        omnipose_logger.info(f'niter: {niter}, interp: {interp}, suppress: {suppress}, calc_trace: {calc_trace}')

    if niter is None:
        niter = 200

    niter = np.uint32(niter)
    cell_px = (Ellipsis,) + tuple(inds)

    flow_pred = torch.tensor(dP, device=device).unsqueeze(0)
    shape = flow_pred.shape
    B = shape[0]
    dim = shape[1]
    dims = shape[-dim:]

    coords = [torch.arange(0, l, device=device) for l in dims]
    mesh = torch.meshgrid(coords, indexing="ij")
    init_shape = [B, 1] + ([1] * len(dims))
    initial_points = torch.stack(mesh, dim=0)
    initial_points = initial_points.repeat(init_shape).float()

    final_points = initial_points.clone()

    if inds.ndim < 2 or inds.shape[0] < dim:
        omnipose_logger.warning('WARNING: no mask pixels found')
        tr = None
    else:
        final_p, tr = steps_batch(initial_points[cell_px],
                                  flow_pred,
                                  niter,
                                  omni=omni,
                                  suppress=suppress,
                                  interp=interp,
                                  calc_trace=calc_trace,
                                  verbose=verbose)

        final_points[cell_px] = final_p.squeeze()

    p = final_points.squeeze().cpu().numpy()
    if verbose:
        omnipose_logger.info('done follow_flows')
    return p, inds, tr


def remove_bad_flow_masks(masks, flows, coords=None, affinity_graph=None, threshold=0.4, use_gpu=False, device=None, omni=True):
    """Remove masks which have inconsistent flows."""
    merrors, _ = flow_error(masks, flows, coords, affinity_graph, use_gpu, device, omni)
    badi = 1 + (merrors > threshold).nonzero()[0]
    masks[np.isin(masks, badi)] = 0
    return masks


def _meshgrid(shape):
    ranges = [np.arange(dim) for dim in shape]
    return np.meshgrid(*ranges, indexing='ij')


def flow_error(maski, dP_net, coords=None, affinity_graph=None, use_gpu=False, device=None, omni=True):
    """Error in flows from predicted masks vs flows predicted by network run on image."""
    if dP_net.shape[1:] != maski.shape:
        omnipose_logger.info('ERROR: net flow is not same size as predicted masks')
        return

    fastremap.renumber(maski, in_place=True)

    idx = -1
    dim = maski.ndim
    dP_masks = masks_to_flows(maski, dim=dim, coords=coords, affinity_graph=affinity_graph,
                              use_gpu=use_gpu, device=device, omni=omni)[idx].cpu().numpy()
    flow_errors = np.zeros(maski.max())

    for i in range(dP_masks.shape[0]):
        flow_errors += mean((dP_masks[i] - dP_net[i] / 5.) ** 2, maski,
                            index=np.arange(1, maski.max() + 1))

    return flow_errors, dP_masks



def get_masks_cp(p, iscell=None, rpad=20, flows=None, use_gpu=False, device=None):
    """ create masks using pixel convergence after running dynamics
    
    Makes a histogram of final pixel locations p, initializes masks 
    at peaks of histogram and extends the masks from the peaks so that
    they include all pixels with more than 2 final pixels p. Discards 
    masks with flow errors greater than the threshold. 

    Parameters
    ----------------
    p: float32, 3D or 4D array
        final locations of each pixel after dynamics,
        size [axis x Ly x Lx] or [axis x Lz x Ly x Lx].
    iscell: bool, 2D or 3D array
        if iscell is not None, set pixels that are 
        iscell False to stay in their original location.
    rpad: int (optional, default 20)
        histogram edge padding
    flows: float, 3D or 4D array (optional, default None)
        flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]. If flows
        is not None, then masks with inconsistent flows are removed using 
        `remove_bad_flow_masks`.

    Returns
    ---------------
    M0: int, 2D or 3D array
        masks with inconsistent flow masks removed, 
        0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]
    
    """
    pflows = []
    edges = []
    shape0 = p.shape[1:]
    dims = len(p)
    if iscell is not None:
        if dims==3:
            inds = np.meshgrid(np.arange(shape0[0]), np.arange(shape0[1]),
                np.arange(shape0[2]), indexing='ij')
        elif dims==2:
            inds = np.meshgrid(np.arange(shape0[0]), np.arange(shape0[1]),
                     indexing='ij')
        for i in range(dims):
            p[i, ~iscell] = inds[i][~iscell]
    
    for i in range(dims):
        pflows.append(p[i].flatten().astype('int32'))
        edges.append(np.arange(-.5-rpad, shape0[i]+.5+rpad, 1))

    h, _ = np.histogramdd(pflows, bins=edges)
    hmax = h.copy()
    for i in range(dims):
        hmax = maximum_filter1d(hmax, 5, axis=i)

    seeds = np.nonzero(np.logical_and(h-hmax>-1e-6, h>10))
    Nmax = h[seeds]
    isort = np.argsort(Nmax)[::-1]
    for s in seeds:
        s = s[isort]
    pix = list(np.array(seeds).T)

    shape = h.shape
    if dims==3:
        expand = np.nonzero(np.ones((3,3,3)))
    else:
        expand = np.nonzero(np.ones((3,3)))
    for e in expand:
        e = np.expand_dims(e,1)

    for iter in range(5):
        for k in range(len(pix)):
            if iter==0:
                pix[k] = list(pix[k])
            newpix = []
            iin = []
            for i,e in enumerate(expand):
                epix = e[:,np.newaxis] + np.expand_dims(pix[k][i], 0) - 1
                epix = epix.flatten()
                iin.append(np.logical_and(epix>=0, epix<shape[i]))
                newpix.append(epix)
            iin = np.all(tuple(iin), axis=0)
            for p in newpix:
                p = p[iin]
            newpix = tuple(newpix)
            igood = h[newpix]>2
            for i in range(dims):
                pix[k][i] = newpix[i][igood]
            if iter==4:
                pix[k] = tuple(pix[k])
    
    M = np.zeros(h.shape, np.int32)
    for k in range(len(pix)):
        M[pix[k]] = 1+k
        
    for i in range(dims):
        pflows[i] = pflows[i] + rpad
    M0 = M[tuple(pflows)]
    
    # remove big masks
    _,counts = np.unique(M0, return_counts=True)
    big = np.prod(shape0) * 0.4
    for i in np.nonzero(counts > big)[0]:
        M0[M0==i] = 0
    _,M0 = np.unique(M0, return_inverse=True)
    M0 = np.reshape(M0, shape0)

    # moved to compute masks
    # if M0.max()>0 and threshold is not None and threshold > 0 and flows is not None:
    #     M0 = remove_bad_flow_masks(M0, flows, threshold=threshold, use_gpu=use_gpu, device=device)
    #     _,M0 = np.unique(M0, return_inverse=True)
    #     M0 = np.reshape(M0, shape0).astype(np.int32)

    return M0
