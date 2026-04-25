from .imports import *

from ocdkit.measure import (
    label_overlap as _label_overlap,
    intersection_over_union as _intersection_over_union,
    true_positive as _true_positive,
)


def _circle_mask(radius):
    rr = np.arange(-radius, radius + 1)
    yy, xx = np.meshgrid(rr, rr, indexing="ij")
    rs = np.sqrt(yy ** 2 + xx ** 2)
    return rs, yy, xx


def mask_ious(masks_true, masks_pred):
    """ return best-matched masks """
    iou = _intersection_over_union(masks_true, masks_pred)[1:,1:]
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= 0.5).astype(float) - iou / (2*n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
    iout = np.zeros(masks_true.max())
    iout[true_ind] = iou[true_ind,pred_ind]
    preds = np.zeros(masks_true.max(), 'int')
    preds[true_ind] = pred_ind+1
    return iout, preds

def boundary_scores(masks_true, masks_pred, scales):
    """ boundary precision / recall / Fscore """
    diams = [core.diam.diameters(lbl) for lbl in masks_true]
    precision = np.zeros((len(scales), len(masks_true)))
    recall = np.zeros((len(scales), len(masks_true)))
    fscore = np.zeros((len(scales), len(masks_true)))
    for j, scale in enumerate(scales):
        for n in range(len(masks_true)):
            diam = max(1, scale * diams[n])
            r = int(np.ceil(diam))
            rs, ys, xs = _circle_mask(r)
            filt = (rs <= diam).astype(np.float32)
            otrue = masks_to_outlines(masks_true[n])
            otrue = convolve(otrue, filt)
            opred = masks_to_outlines(masks_pred[n])
            opred = convolve(opred, filt)
            tp = np.logical_and(otrue==1, opred==1).sum()
            fp = np.logical_and(otrue==0, opred==1).sum()
            fn = np.logical_and(otrue==1, opred==0).sum()
            precision[j,n] = tp / (tp + fp)
            recall[j,n] = tp / (tp + fn)
        fscore[j] = 2 * precision[j] * recall[j] / (precision[j] + recall[j])
    return Result(precision=precision, recall=recall, fscore=fscore)


def aggregated_jaccard_index(masks_true, masks_pred):
    """ AJI = intersection of all matched masks / union of all masks 
    
    Parameters
    ------------
    
    masks_true: list of ND-arrays (int) or ND-array (int) 
        where 0=NO masks; 1,2... are mask labels
    masks_pred: list of ND-arrays (int) or ND-array (int) 
        ND-array (int) where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    aji : aggregated jaccard index for each set of masks

    """

    aji = np.zeros(len(masks_true))
    for n in range(len(masks_true)):
        iout, preds = mask_ious(masks_true[n], masks_pred[n])
        inds = np.arange(0, masks_true[n].max(), 1, int)
        overlap = _label_overlap(masks_true[n], masks_pred[n])
        union = np.logical_or(masks_true[n]>0, masks_pred[n]>0).sum()
        overlap = overlap[inds[preds>0]+1, preds[preds>0].astype(int)]
        aji[n] = overlap.sum() / union
    return aji 


def average_precision(masks_true, masks_pred, threshold=[0.5, 0.75, 0.9]):
    """ average precision estimation: AP = TP / (TP + FP + FN)

    This function is based heavily on the *fast* stardist matching functions
    (https://github.com/mpicbg-csbd/stardist/blob/master/stardist/matching.py)

    Parameters
    ------------
    
    masks_true: list of ND-arrays (int) or ND-array (int) 
        where 0=NO masks; 1,2... are mask labels
    masks_pred: list of ND-arrays (int) or ND-array (int) 
        ND-array (int) where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    ap: array [len(masks_true) x len(threshold)]
        average precision at thresholds
    tp: array [len(masks_true) x len(threshold)]
        number of true positives at thresholds
    fp: array [len(masks_true) x len(threshold)]
        number of false positives at thresholds
    fn: array [len(masks_true) x len(threshold)]
        number of false negatives at thresholds

    """
    not_list = False
    if not isinstance(masks_true, list):
        masks_true = [masks_true]
        masks_pred = [masks_pred]
        not_list = True
    if not isinstance(threshold, list) and not isinstance(threshold, np.ndarray):
        threshold = [threshold]
    ap  = np.zeros((len(masks_true), len(threshold)), np.float32)
    tp  = np.zeros((len(masks_true), len(threshold)), np.float32)
    fp  = np.zeros((len(masks_true), len(threshold)), np.float32)
    fn  = np.zeros((len(masks_true), len(threshold)), np.float32)
    n_true = np.array(list(map(np.max, masks_true)))
    n_pred = np.array(list(map(np.max, masks_pred)))
#     if len(n_pred) < 1:
#         n_pred = [0]
    for n in range(len(masks_true)):
        #_,mt = np.reshape(np.unique(masks_true[n], return_index=True), masks_pred[n].shape)
        if n_pred[n] > 0:
            iou = _intersection_over_union(masks_true[n], masks_pred[n])[1:, 1:]
            for k,th in enumerate(threshold):
                tp[n,k] = _true_positive(iou, th)
        fp[n] = n_pred[n] - tp[n]
        fn[n] = n_true[n] - tp[n]
        ap[n] = tp[n] / (tp[n] + fp[n] + fn[n]) # this is the jaccard index, not precision, right? 
        # this is tp[n] / (tp[n] + n_pred[n] - tp[n] + n_true[n] - tp[n]) = tp[n] / ( n_pred[n] + n_true[n] - tp[n])
        
    if not_list:
        ap, tp, fp, fn = ap[0], tp[0], fp[0], fn[0]
    return Result(ap=ap, tp=tp, fp=fp, fn=fn)

def flow_error(maski, dP_net, use_gpu=False, device=None):
    """ error in flows from predicted masks vs flows predicted by network run on image

    This function serves to benchmark the quality of masks, it works as follows
    1. The predicted masks are used to create a flow diagram
    2. The mask-flows are compared to the flows that the network predicted

    If there is a discrepancy between the flows, it suggests that the mask is incorrect.
    Masks with flow_errors greater than 0.4 are discarded by default via flow_threshold.

    Parameters
    ------------
    
    maski: ND-array (int) 
        masks produced from running dynamics on dP_net, 
        where 0=NO masks; 1,2... are mask labels
    dP_net: ND-array (float) 
        ND flows where dP_net.shape[1:] = maski.shape

    Returns
    ------------

    flow_errors: float array with length maski.max()
        mean squared error between predicted flows and flows from masks
    dP_masks: ND-array (float)
        ND flows produced from the predicted masks
    
    """
    if dP_net.shape[1:] != maski.shape:
        print('ERROR: net flow is not same size as predicted masks')
        return

    # flows predicted from estimated masks
    dP_masks = core.masks_to_flows(maski, use_gpu=use_gpu, device=device)
    # difference between predicted flows vs mask flows
    flow_errors=np.zeros(maski.max())
    for i in range(dP_masks.shape[0]):
        flow_errors += mean((dP_masks[i] - dP_net[i]/5.)**2, maski,
                            index=np.arange(1, maski.max()+1))

    return flow_errors, dP_masks
