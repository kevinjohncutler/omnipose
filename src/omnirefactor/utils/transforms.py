from .imports import *

# No reason to support anything but pytorch for omnipose
# I want it to fail here otherwise, much easier to debug 
import torch
TORCH_ENABLED = True 

### This section defines the tiling functions 
def get_module(x):
    if isinstance(x, (np.ndarray, tuple, int, float, da.Array)) or np.isscalar(x):
        return np
    elif torch.is_tensor(x):
        return torch
    else:
        raise ValueError("Input must be a numpy array, a tuple, a torch tensor, an integer, or a float")

def safe_divide(num, den, cutoff=0):
    """ Division ignoring zeros and NaNs in the denominator.""" 
    module = get_module(num)
    valid_den = (den > cutoff) & module.isfinite(den) #isfinite catches both nan and inf

    if module == np:
        r = num.astype(np.float32, copy=False)
        r = np.divide(r, den, out=np.zeros_like(r), where=valid_den)
    elif module == torch:
        r = num.float()
        den = den.float()
        small_val = torch.finfo(den.dtype).tiny  # smallest positive representable number
        safe_den = torch.where(valid_den, den, small_val)
        r = torch.div(r, safe_den)
    else:
        raise TypeError("num must be a numpy array or a PyTorch tensor")

    return r

def rescale(T, floor=None, ceiling=None, exclude_dims=None):
    module = get_module(T)
    if exclude_dims is not None:
        if isinstance(exclude_dims, int):
            exclude_dims = (exclude_dims,)
        axes = tuple(i for i in range(T.ndim) if i not in exclude_dims)
        newshape = [T.shape[i] if i in exclude_dims else 1 for i in range(T.ndim)]
    else:
        axes = None
        newshape = T.shape  # If no axes are excluded, keep the original shape
    
    if ceiling is None:
        ceiling = module.amax(T, axis=axes)
        if exclude_dims is not None:
            ceiling = ceiling.reshape(*newshape)
    if floor is None:
        floor = module.amin(T, axis=axes)
        if exclude_dims is not None:
            floor = floor.reshape(*newshape)
            
    T = safe_divide(T - floor, ceiling - floor)

    return T



def normalize_field(mu,use_torch=False,cutoff=0):
    """ normalize all nonzero field vectors to magnitude 1
    
    Parameters
    ----------
    mu: ndarray, float
        Component array of lenth N by L1 by L2 by ... by LN. 
    
    Returns
    --------------
    normalized component array of identical size. 
    """
    if use_torch:
        mag = torch_norm(mu,dim=0)
        # out = torch.zeros_like(mu)
        # sel = mag>cutoff
        # out[:,sel] = torch.div(mu[:,sel],mag[sel])
        # return out
        # return torch.where(mag>cutoff,mu/mag,torch.zeros_like(mu))
        return torch.where(mag>cutoff,mu/mag,mu)
        
    else:
        mag = np.sqrt(np.nansum(mu**2,axis=0))
        return safe_divide(mu,mag,cutoff)
    
        
# @torch.jit.script
def torch_norm(a,dim=0,keepdim=False):
    """ Wrapper for torch.linalg.norm to handle ARM architecture. """
    # if ARM: 
    #     #torch.linalg.norm not implemented on MPS yet
    #     # this is the fastest I have tested but still slow in comparison 
    #     return a.square().sum(dim=dim,keepdim=keepdim).sqrt()
    # else:
    #     return torch.linalg.norm(a,dim=dim,keepdim=keepdim)
        
        
        # Compute squared norm with a minimal number of intermediate tensors.
    norm_sq = (a * a).sum(dim=dim, keepdim=keepdim)
    # Use the in-place sqrt when possible (if not tracking gradients).
    return norm_sq.sqrt_() if not norm_sq.requires_grad else norm_sq.sqrt()
    
    # in the future when MPS supports it, just use try catch and print a warning to upgrade torch 






def bin_counts(data, num_bins=256):
    """Compute the counts of values in bins.

    Parameters:
    data (np.ndarray): Input data.
    num_bins (int): Number of bins.

    Returns:
    np.ndarray: Counts of values in each bin.
    """
    unique_values, counts = fastremap.unique(data, return_counts=True) # this only works on integer, e.g. raw images 
    bin_edges = np.linspace(unique_values.min(), unique_values.max(), num_bins+1)
    # bin_indices = np.digitize(unique_values, bin_edges)
    # binned_counts = np.bincount(bin_indices, weights=counts, minlength=num_bins+1)


    bin_indices = np.digitize(unique_values, bin_edges) - 1
    binned_counts = np.bincount(bin_indices, weights=counts, minlength=num_bins)

    # print(binned_counts.shape, bin_edges.shape)
    bin_start = bin_edges[:-1]

    # Ensure the shapes match
    binned_counts = binned_counts[:-1]
    
    return binned_counts, bin_start
    

from scipy.stats import gaussian_kde
def compute_density(x, y, bw_method=None):
    """Compute the density of points along a curve.

    Parameters:
    x (np.ndarray): x-coordinates of the points on the curve.
    y (np.ndarray): y-coordinates of the points on the curve.

    Returns:
    np.ndarray: Density of the points along the curve.
    """
    # Combine the x and y coordinates into a 2D array
    points = np.vstack([x, y])

    # Compute the KDE for the original points
    kde = gaussian_kde(points,bw_method=bw_method)
    density = kde(points)

    # Compute the KDE for the inverted points
    inverted_points = np.vstack([-x, y])
    inverted_kde = gaussian_kde(inverted_points,bw_method=bw_method)
    inverted_density = inverted_kde(inverted_points)

    # Take the average of the two densities
    symmetric_density = (density + inverted_density) / 2
    symmetric_density = rescale(symmetric_density)

    return symmetric_density
    
    
def qnorm(Y, 
            nbins=100,
            bw_method=2, 
            density_cutoff=None, 
            density_quantile=[.001,.999],
            debug=False, 
            dx = None,
            log=False,
            eps=1):

    if dx is not None:
        X = Y[:,::dx,::dx]
    else:
        X = Y
        
    # make it into an integer form that fasrtremap can work on  
    if X.dtype not in [np.uint8,np.uint16,np.uint32,np.uint64]:
        X = to_16_bit(X)
    counts, unique = bin_counts(X,nbins)
    # print('uu',np.std(unique)/np.mean(unique)) # curious this is the same for all images at same nbin 
    sel = counts>0
    counts = counts[sel]
    unique = unique[sel]
    x = np.arange(len(counts))
    if log:
        # x = np.log(unique+(unique==0))
        # y = np.log(counts+(counts==0))
        # x = np.log(unique+eps)
        y = np.log(counts+eps)
    else:
        y = counts
    
    d = compute_density(x,y,bw_method=bw_method)
    
    if not isinstance(density_quantile,list):
        density_quantile = [density_quantile,density_quantile]
    
    if density_cutoff is None:
        density_cutoff = np.quantile(d,density_quantile) 
        if debug: 
            print('dc',density_cutoff)
    elif not isinstance(density_cutoff,list):
        density_cutoff = [density_cutoff,density_cutoff]
        
    imin = np.argwhere(d>density_cutoff[0])[0][0]
    imax = np.argwhere(d>density_cutoff[1])[-1][0]
    vmin, vmax = unique[imin], unique[imax]
    
    if vmax>vmin:
        scale_factor = np.float16(1.0 / (vmax - vmin))
        # r = ne.evaluate('Y * scale_factor')
        # ne.evaluate("where(r > 1, 1, r)", out=r) 
        r = ne.evaluate('where(X * scale_factor > 1, 1, X * scale_factor)')
    else:
        r = X

    if debug:
        return r, x, y, d, imin, imax, vmin, vmax
    else:
        return r
        
        
# should add an option for foreground/background to allow upper to refer to foreground 
# and lower to background
def normalize99(Y, lower=0.01, upper=99.99, contrast_limits=None, dim=None):
    """ normalize array/tensor so 0.0 is 0.01st percentile and 1.0 is 99.99th percentile 
    Upper and lower percentile ranges configurable. 
    
    Parameters
    ----------
    Y: ndarray/tensor, float
        Input array/tensor. 
    upper: float
        upper percentile above which pixels are sent to 1.0
    
    lower: float
        lower percentile below which pixels are sent to 0.0
        
    contrast_limits: list, float (optional, override computation)
        list of two floats, lower and upper contrast limits
    
    Returns
    --------------
    normalized array/tensor with a minimum of 0 and maximum of 1
    
    """
    module = get_module(Y)
    
    if contrast_limits is None:
    
        quantiles = np.array([lower, upper]) / 100
        if module == torch:
            quantiles = torch.tensor(quantiles, dtype=Y.dtype, device=Y.device)
   
        if dim is not None:
            # Reshape Y into a 2D tensor for quantile computation
            Y_flattened = Y.reshape(Y.shape[dim], -1)

            lower_val, upper_val = module.quantile(Y_flattened, quantiles, axis=-1)        
            
            # Reshape back into original shape for broadcasting
            if dim == 0:
                lower_val = lower_val.reshape(Y.shape[dim], *([1] * (len(Y.shape) - 1)))
                upper_val = upper_val.reshape(Y.shape[dim], *([1] * (len(Y.shape) - 1)))
            else:
                lower_val = lower_val.reshape(*Y.shape[:dim], *([1] * (len(Y.shape) - dim - 1)))
                upper_val = upper_val.reshape(*Y.shape[:dim], *([1] * (len(Y.shape) - dim - 1)))
        else:
            # lower_val, upper_val = module.quantile(Y, quantiles)
            try:
                lower_val, upper_val = module.quantile(Y, quantiles)
            except RuntimeError:
                lower_val, upper_val = auto_chunked_quantile(Y, quantiles)

    else:
        if module == np:
            contrast_limits = np.array(contrast_limits)
        elif module == torch:
            contrast_limits = torch.tensor(contrast_limits)

        lower_val, upper_val = contrast_limits
        
    
    # Y = module.clip(Y, lower_val, upper_val) # is this needed? 
    # Y -= lower_val
    # Y /= (upper_val - lower_val)
    
    # return Y
    # return (Y-lower_val)/(upper_val-lower_val)
    # return module.clip((Y-lower_val)/(upper_val-lower_val),0,1)
    # return module.clip((Y-lower_val)/(upper_val-lower_val),0,1)
    # in this case, since lower_val is not the absolute minimum, but the lowerr quanitle, 
    # Y-lower_val can be less than zero. Likewise for the upward scalimg being slightly >1. 
    return module.clip(safe_divide(Y-lower_val,upper_val-lower_val),0,1)


def searchsorted(tensor, value):
    """Find the indices where `value` should be inserted in `tensor` to maintain order."""
    return (tensor < value).sum()


def compute_quantiles(sorted_array, lower=0.01, upper=0.99):
    """Compute a pair of quantiles of a sorted array.

    Parameters:
    sorted_array (np.ndarray): Input array sorted in ascending order.
    lower (float): Lower quantile to compute, which must be between 0 and 1 inclusive.
    upper (float): Upper quantile to compute, which must be between 0 and 1 inclusive.

    Returns:
    tuple: The lower and upper quantiles of the input array.
    """
    assert 0 <= lower <= 1, "Lower quantile must be between 0 and 1"
    assert 0 <= upper <= 1, "Upper quantile must be between 0 and 1"
    lower_index = int(lower * (len(sorted_array) - 1))
    upper_index = int(upper * (len(sorted_array) - 1))
    return sorted_array[lower_index], sorted_array[upper_index]

def quantile_rescale(Y, lower=0.0001, upper=0.9999, contrast_limits=None, bins=None):
   
    sorted_array = np.sort(Y.flatten(),kind='mergesort')
    lower_val, upper_val  = compute_quantiles(sorted_array, lower, upper)
    
    # return np.clip((Y - lower_val) / (upper_val - lower_val), 0, 1)
    
    # return np.clip(safe_divide(Y - lower_val, upper_val - lower_val), 0, 1)
    r = safe_divide(Y - lower_val, upper_val - lower_val)
    r [r<0] = 0
    r [r>1] = 1
    return r
    

    
    

def normalize99_hist(Y, lower=0.01, upper=99.99, contrast_limits=None, bins=None):
    """ normalize array/tensor using 1% and 99% quantiles
    
    Parameters
    ----------
    Y: ndarray/tensor, float
        Input array/tensor. 
    contrast_limits: list of float
        The lower and upper quantiles to use for normalization. Default is [0.01, 0.99].
    bins: int
        The number of bins to use for the histogram. Default is 1000.
    
    Returns
    --------------
    normalized array/tensor with values between 0 and 1
    
    """
    upper = upper/100
    lower = lower/100
    
    module = get_module(Y)
    if bins is None:
        if module == np:
            num_elements = Y.size
        elif module == torch:
            num_elements = Y.numel()
        bins = int(np.sqrt(num_elements))
        # bins = int(num_elements)
            
    # print(bins,num_elements,'bbv')
    if contrast_limits is None:
        # Estimate the quantiles using a histogram
        # if module == np:
        # elif module == torch:
        #     hist = torch.histc(Y, bins=bins)
        #     bin_edges = torch.linspace(Y.min(), Y.max(), steps=bins+1)

        hist, bin_edges = module.histogram(Y,bins=bins)
        # print(len(bin_edges))
        
        cdf = module.cumsum(hist, axis=0) / module.sum(hist)
        lower_val = bin_edges[searchsorted(cdf, lower)]
        upper_val = bin_edges[searchsorted(cdf, upper)]
    else:
        if module == np:
            contrast_limits = np.array(contrast_limits)
        elif module == torch:
            contrast_limits = torch.tensor(contrast_limits)

        lower_val, upper_val = contrast_limits
        
    # Normalize Y to the range [0, 1]
    # Y_normalized = module.clip((Y - lower_val) / (upper_val - lower_val), 0, 1)
    r = safe_divide(Y - lower_val, upper_val - lower_val)
    r [r<0] = 0
    r [r>1] = 1
    return r
    
    

# lol silent p, p-norm pun 
def pnormalize(Y, p_min=-1,p_max = 10):
    """ normalize array/tensor using p-norm
    
    Parameters
    ----------
    Y: ndarray/tensor, float
        Input array/tensor. 
    p: float
        The p value for the p-norm. Default is 2.
    
    Returns
    --------------
    normalized array/tensor with p-norm of 1
    
    """
    
    module = get_module(Y)
    
    # Compute the p-norm
    # upper_val = module.linalg.norm(Y, p_max)
    # lower_val = module.linalg.norm(Y, p_min)
    lower_val = (module.abs(Y*1.0)**p_min).sum()**(1./p_min)
    upper_val = (module.abs(Y*1.0)**p_max).sum()**(1./p_max)
        
    # print(upper_val,lower_val)
    
    return module.clip(safe_divide(Y-lower_val,upper_val-lower_val),0,1)



def auto_chunked_quantile(tensor, q):
    # Determine the maximum number of elements that can be handled by PyTorch's quantile function
    max_elements = 16e6 - 1  

    # Determine the number of elements in the tensor
    num_elements = tensor.nelement()

    # Determine the chunk size
    chunk_size = math.ceil(num_elements / max_elements)
    
    # Split the tensor into chunks
    chunks = torch.chunk(tensor, chunk_size)

    # Compute the quantile for each chunk
    return torch.stack([torch.quantile(chunk, q) for chunk in chunks]).mean(dim=0)

try:
    import numexpr as ne
except:
    pass
# from skimage.measure import label, regionprops_table

def normalize_image(im, mask, target=0.5, foreground=False, 
                    iterations=1, scale=1, channel_axis=0, per_channel=True):
    """
    Normalize image by rescaling from 0 to 1 and then adjusting gamma to bring 
    average background to specified value (0.5 by default). This is what I call
    semantic gamma normalization.
    
    Parameters
    ----------
    im: ndarray, float
        input image or volume
        
    mask: ndarray, int or bool
        input labels or foreground mask
    
    target: float
        target background/foreground value in the range 0-1
    
    channel_axis: int
        the axis that contains the channels
    
    Returns
    --------------
    gamma-normalized array with a minimum of 0 and maximum of 1
    
    """
    # im = rescale(im) * scale
    # im = rescale(im).astype('float32') * scale
    im = im.astype('float32') * scale
    im_min = im.min()
    im_max = im.max()
    ne.evaluate("(im - im_min) / (im_max - im_min)",out=im)
    
    if im.ndim > 2:  # assume last axis is channel axis
        im = np.moveaxis(im, channel_axis, -1)  # move channels to last axis
    else:
        im = np.expand_dims(im, axis=-1)
        
    if not isinstance(mask, list):
        mask = np.expand_dims(mask, axis=-1)  # Add a new axis to mask
        mask = np.broadcast_to(mask, im.shape)  # Broadcast mask to the shape of im
        
    # for k in range(len(mask)):
    #     bin0 = binary_erosion(mask[k]>0 if foreground else mask[k] == 0, iterations=iterations) 
    #     source_target = np.mean(im[k][bin0])
    #     im[k] = im[k] ** (np.log(target) / np.log(source_target))
        
    bin0 = mask>0 if foreground else mask == 0
    if iterations > 0:
        # Create a structuring element that erodes only along the last two dimensions
        structure = np.ones((3,) * (im.ndim - 1) + (1,))
        structure[1, ...] = 0
        bin0 =  binary_erosion(bin0, structure=structure, iterations=iterations)
        
    # masked_im = np.ma.masked_array(im, mask=np.logical_not(bin0))
    # # source_target = np.ma.mean(masked_im, axis=(0,1) if per_channel else None) 
    # masked_im = im.copy()
    # masked_im[~bin0] = np.nan  # Replace masked values with nan

    # if per_channel:
    #     source_target = np.empty(im.shape[-1])  # Initialize array for mean values
    #     for i in range(im.shape[-1]):
    #         source_target[i] = np.nanmean(masked_im[..., i])
    # else:
    #     source_target = np.nanmean(masked_im)
    
    # Create a mask for the background
    # background_mask = ~bin0

    # Apply the mask to the image
    # masked_im = im.copy()
    # masked_im[bin0] = np.nan  # Replace background values with nan

    # # Compute the mean of the background values along the channel axis
    # source_target = np.apply_along_axis(np.nanmean, -1, masked_im)
        

    masked_im = im.copy()
    masked_im[~bin0] = np.nan
    source_target = np.nanmean(masked_im, axis=(0,1) if per_channel else None)
    source_target = source_target.astype('float32')
    target = np.array(target).astype('float32')
    # print(np.log(source_target).max(),'ss')
    # im = im ** (np.log(target) / np.log(source_target))
    # im **= (np.log(target) / np.log(source_target))
    ne.evaluate("im ** (log(target) / log(source_target))", out=im)
    # im = np.exp(np.log(im+1e-8) * np.log(target) / (np.log(source_target)))    
    # im = np.power(im,np.log(target) / np.log(source_target))
    return np.moveaxis(im, -1, channel_axis).squeeze()



def adjust_contrast_masked(
    img: np.ndarray,
    masks: np.ndarray,
    r_target: float = 1.10,
    plo: float = .01,
    phi: float = 99.99,
    clip_output: bool = True,
):
    """
    Single-call masked contrast adjustment.
    - Anchors a,b from background/foreground percentiles.
    - Applies gamma so mean_fg/mean_bg ≈ r_target on the anchored image.
    - Identity when current ratio already ≈ r_target or when dynamic range collapses.

    Returns (img_out, gamma, anchors)
    """
    x = np.asarray(img, dtype=np.float32)
    m = np.asarray(masks).astype(bool)
    bg = ~m
    fg = m

    # Fallback if masks are degenerate
    if fg.sum() == 0 or bg.sum() == 0:
        return x.copy(), 1.0, (float(np.min(x)), float(np.max(x)))

    # Mask-aware anchors: low from bg, high from fg
    a = np.percentile(x[bg], plo)
    b = np.percentile(x[fg], phi)
    if not np.isfinite(a) or not np.isfinite(b) or b <= a:
        # Fallback anchors if percentiles are ill-posed
        a = float(np.min(x))
        b = float(np.max(x) + 1e-12)

    # Anchor-normalize to [0,1]
    j = (x - a) / (b - a)
    j = np.clip(j, 0.0, 1.0)

    # Current fg/bg mean ratio on anchored image
    m_fg = float(j[fg].mean())
    m_bg = float(j[bg].mean() + 1e-12)
    r = m_fg / m_bg

    # If r_target direction conflicts with data, coerce toward identity
    # Example: if r < 1 but r_target > 1, or vice versa, use identity.
    if (r >= 1.0 and r_target < 1.0) or (r <= 1.0 and r_target > 1.0):
        return j.copy(), 1.0, (a, b)

    # If already close, identity
    if abs(np.log(max(r, 1e-12))) < 1e-8 or abs((r - r_target) / max(r_target, 1e-12)) < 1e-3:
        y = j
        gamma = 1.0
    else:
        # Closed-form gamma on anchored intensities
        gamma = float(np.log(max(r_target, 1e-12)) / np.log(max(r, 1e-12)))
        # Clamp gamma to sane bounds
        gamma = float(np.clip(gamma, 0.2, 5.0))
        y = np.power(j, gamma)

    if clip_output:
        y = np.clip(y, 0.0, 1.0)

    return y.astype(np.float32), gamma, (float(a), float(b))

import torch
from scipy.ndimage import binary_erosion

def gamma_normalize(im, mask, target=1.0, scale=1.0, foreground=True, iterations=0, per_channel=True, channel_axis=-1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    im = rescale(im) * scale
    if im.ndim > 2:  # assume last axis is channel axis
        im = np.moveaxis(im, channel_axis, -1)  # move channels to last axis
    else:
        im = np.expand_dims(im, axis=-1)
        
    if not isinstance(mask, list):
        mask = np.stack([mask] * im.shape[-1], axis=-1)
    im = torch.from_numpy(im).float().to(device)
    mask = torch.from_numpy(mask).float().to(device)


    bin0 = mask > 0 if foreground else mask == 0
    if iterations > 0:
        # Create a structuring element that erodes only along the last two dimensions
        structure = torch.ones((3,) * (im.ndim - 1) + (1,)).to(device)
        structure[1, ...] = 0
        bin0 = torch.from_numpy(binary_erosion(bin0.cpu().numpy(), structure=structure.cpu().numpy(), iterations=iterations)).to(device)

    masked_im = im.masked_fill(~bin0, float('nan'))
    source_target = torch.nanmean(masked_im, dim=(0,1) if per_channel else None)
    im **= (torch.log(target) / torch.log(source_target))

    return im.permute(*[channel_axis] + [i for i in range(im.ndim) if i != channel_axis]).squeeze().cpu().numpy()
    

def localnormalize(im,sigma1=2,sigma2=20):
    im = normalize99(im)
    blur1 = gaussian_filter(im,sigma=sigma1)
    num = im - blur1
    blur2 = gaussian_filter(num*num, sigma=sigma2)
    den = np.sqrt(blur2)
    
    return normalize99(num/den+1e-8)
    
import torchvision.transforms.functional as TF
def localnormalize_GPU(im, sigma1=2, sigma2=20):
    im = normalize99(im)
    kernel_size1 = round(sigma1 * 6)
    kernel_size1 += kernel_size1 % 2 == 0
    blur1 = TF.gaussian_blur(im, kernel_size1, sigma1)
    num = im - blur1
    kernel_size2 = round(sigma2 * 6)
    kernel_size2 += kernel_size2 % 2 == 0
    blur2 = TF.gaussian_blur(num*num, kernel_size2, sigma2)
    den = torch.sqrt(blur2)

    return normalize99(num/den+1e-8)



def rotate(V,theta,order=1,output_shape=None,center=None):
    
    dim = V.ndim
    v1 = np.array([0]*(dim-1)+[1])
    v2 = np.array([0]*(dim-2)+[1,0])

    s_in = V.shape
    if output_shape is None:
        s_out = s_in
    else:
        s_out = output_shape
    M = mgen.rotation_from_angle_and_plane(np.pi/2-theta,v2,v1)
    if center is None:
        c_in = 0.5 * np.array(s_in) 
    else:
        c_in = center
    c_out = 0.5 * np.array(s_out)
    offset = c_in  - np.dot(np.linalg.inv(M), c_out)
    V_rot = affine_transform(V, np.linalg.inv(M), offset=offset, 
                                           order=order, output_shape=output_shape)

    return V_rot

