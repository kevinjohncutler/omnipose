import os
import numpy as np
from scipy.ndimage import gaussian_filter
from . import utils, io, transforms
from omnipose.utils import rescale
from omnipose.plot import colorize, figure

try:
    import matplotlib
    MATPLOTLIB_ENABLED = True 
except:
    MATPLOTLIB_ENABLED = False

try:
    from skimage import color
    from skimage.segmentation import find_boundaries
    SKIMAGE_ENABLED = True 
except:
    SKIMAGE_ENABLED = False

try:
    from omnipose.plot import sinebow
    import ncolor
    OMNI_INSTALLED = True
except:
    OMNI_INSTALLED = False
    
    

import numpy as np
from colour import XYZ_to_sRGB, Oklab_to_XYZ

# ----------------------------------------------------------------------
def _uniform_chroma_at_L(L_vals, cθ, sθ, iters=10, margin=0.97):
    """
    Return, for each lightness in `L_vals`, the largest *uniform* chroma
    every hue can sustain inside sRGB.

    L_vals : (M,) array-like   - candidate OK-Lab lightness values
    cθ,sθ   : (N,)             - cos/sin lookup tables for the hues
    """
    L_vals = np.asarray(L_vals, np.float32)
    M, N   = L_vals.size, cθ.size

    # track chroma bounds **per hue**  → shape (M,N)
    lo = np.zeros((M, N), np.float32)
    hi = np.full((M, N), 0.5, np.float32)

    # repeat L-column N times once; avoids repeat/broadcast in loop
    Lmat = np.repeat(L_vals[:, None], N, axis=1)             # (M,N)

    for _ in range(iters):
        C   = 0.5 * (lo + hi)                                # (M,N)
        Lab = np.stack((Lmat, C * cθ, C * sθ), axis=-1)      # (M,N,3)
        rgb = XYZ_to_sRGB(Oklab_to_XYZ(Lab))
        ok  = np.all((rgb >= 0) & (rgb <= 1), axis=-1)       # (M,N) mask

        lo  = np.where(ok, C, lo)                            # binary search
        hi  = np.where(ok, hi, C)

    return lo.min(axis=1) * margin                           # (M,)

# ----------------------------------------------------------------------
def build_balanced_wheel(N=256, L_range=(0.35, 0.80),
                              coarse=21, fine_step=0.002,
                              margin=0.97, iters=10):
    """
    Band-free OKLCh wheel with deepest common chroma, found by a
    hierarchical (coarse→fine) lightness search.

    Returns
    -------
    wheel : (N,3) ndarray of sRGB in [0,1]
    L_opt : float   chosen OK-Lab lightness
    C_opt : float   chosen uniform chroma
    """
    θ       = np.linspace(0, 2*np.pi, N, endpoint=False, dtype=np.float32)
    cθ, sθ  = np.cos(θ), np.sin(θ)

    # --- coarse sweep ------------------------------------------------------
    L_coarse = np.linspace(*L_range, coarse, dtype=np.float32)
    C_coarse = _uniform_chroma_at_L(L_coarse, cθ, sθ, iters, margin)
    L_best   = float(L_coarse[np.argmax(C_coarse)])

    # --- fine sweep around the best coarse cell ---------------------------
    half = 5 * fine_step                          # ± five steps window
    L_fine = np.arange(L_best - half, L_best + half + fine_step,
                       fine_step, dtype=np.float32)
    C_fine = _uniform_chroma_at_L(L_fine, cθ, sθ, iters, margin)
    j      = int(np.argmax(C_fine))
    L_opt, C_opt = float(L_fine[j]), float(C_fine[j])

    # --- final wheel -------------------------------------------------------
    Lab   = np.stack((np.full(N, L_opt, np.float32),
                      C_opt * cθ, C_opt * sθ), axis=-1)
    wheel = XYZ_to_sRGB(Oklab_to_XYZ(Lab)).astype(np.float32)
    return wheel, L_opt, C_opt


import numpy as np
from colour import XYZ_to_sRGB, Oklab_to_XYZ

# ─────────────────────────────────────────────────────────────────────────────
def build_boundary_wheel(N: int = 256, L0: float = 0.65,
                         margin: float = 0.97, iters: int = 10):
    """
    Constant-L wheel with *per-hue* maximum chroma (no uniform-chroma clamp).

    Parameters
    ----------
    N       - number of discrete hues (samples around the circle)
    L0      - fixed OK-Lab lightness to keep for every colour
    margin  - safety factor (< 1) applied to each hue's Cₘₐₓ(h)
    iters   - bisection iterations; 10 → ~1 × 10⁻³ precision

    Returns
    -------
    wheel   - (N, 3) ndarray, sRGB in [0, 1]
    C_max   - (N,) ndarray, chroma actually used for every hue
    """
    θ      = np.linspace(0.0, 2.0*np.pi, N, endpoint=False, dtype=np.float32)
    cθ, sθ = np.cos(θ), np.sin(θ)

    # Binary-search bounds for chroma, one scalar per hue
    lo = np.zeros(N, np.float32)
    hi = np.full(N, 0.5, np.float32)
    L  = np.full(N, L0, np.float32)

    for _ in range(iters):
        C   = 0.5 * (lo + hi)                          # midpoint
        Lab = np.stack((L, C*cθ, C*sθ), axis=-1)       # (N,3)
        rgb = XYZ_to_sRGB(Oklab_to_XYZ(Lab))
        ok  = np.all((rgb >= 0) & (rgb <= 1), axis=-1) # inside gamut?

        lo  = np.where(ok, C, lo)                      # raise lower bound
        hi  = np.where(ok, hi, C)                      # lower upper bound

    C_max = lo * margin                                # small safety margin
    Lab   = np.stack((L, C_max*cθ, C_max*sθ), axis=-1)
    wheel = XYZ_to_sRGB(Oklab_to_XYZ(Lab)).astype(np.float32)
    return wheel, C_max
    

# ----------------------------------------------------------------------
def build_slice_interpolated_wheel(
        N: int   = 256,
        weight: float = 0,     # 0 → lower boundary, 0.5 → mid-band, 1 → upper
        margin: float = 0.97,
        L_samples: int = 33,     # coarse lightness sweep to bracket Cₘₐₓ(h)
        L_fine: int    = 301,    # fine lightness grid for slice boundaries
        iters: int     = 10,
):
    """
    Uniform-chroma wheel with *varying* lightness:

        1. C₀  = margin · minₕ max_L C_max(h, L)
        2. For each hue, find lowest & highest L where (L, C₀, h) is in-gamut.
        3. Blend those boundaries with `weight`.

    Returns
    -------
    wheel  - (N,3) sRGB colours
    C0     - chosen uniform chroma
    L(h)   - per-hue lightness path actually used
    """
    θ      = np.linspace(0, 2*np.pi, N, endpoint=False, dtype=np.float32)
    cθ, sθ = np.cos(θ), np.sin(θ)

    # -- helper: per-hue C_max at fixed lightness ----------------------------
    def _cmax_per_hue(L0):
        lo = np.zeros(N, np.float32); hi = np.full(N, 0.5, np.float32)
        L  = np.full(N, L0, np.float32)
        for _ in range(iters):
            C   = 0.5*(lo + hi)
            rgb = XYZ_to_sRGB(Oklab_to_XYZ(np.stack((L, C*cθ, C*sθ), -1)))
            ok  = np.all((rgb >= 0) & (rgb <= 1), -1)
            lo, hi = np.where(ok, C, lo), np.where(ok, hi, C)
        return lo                                                       # C_max(h)

    # -- 1. global uniform chroma C₀ ----------------------------------------
    C_max_h = np.zeros(N, np.float32)
    for L0 in np.linspace(0, 1, L_samples, dtype=np.float32):
        C_max_h = np.maximum(C_max_h, _cmax_per_hue(L0))
    C0 = C_max_h.min() * margin

    # -- 2. hue-wise slice boundaries at that C₀ ----------------------------
    L_f = np.linspace(0, 1, L_fine, dtype=np.float32)[:, None]          # (L,1)
    # --- replaces the old Lab = np.stack(...) line ------------------------
    L_mat = np.repeat(L_f, N, axis=1)            # (L_fine, N)
    Cc    = np.broadcast_to(C0 * cθ, L_mat.shape)
    Cs    = np.broadcast_to(C0 * sθ, L_mat.shape)
    Lab   = np.stack((L_mat, Cc, Cs), axis=-1)   # (L_fine, N, 3)
    rgb   = XYZ_to_sRGB(Oklab_to_XYZ(Lab))
    valid = np.all((rgb >= 0) & (rgb <= 1), axis=-1)                    # (L,N)

    L_lo = L_f[np.argmax(valid, axis=0), 0]                             # first True
    L_hi = L_f[::-1][np.argmax(valid[::-1], axis=0), 0]                # last  True

    # -- 3. interpolate with `weight` ---------------------------------------
    # L_path = (1.0 - weight) * L_lo + weight * L_hi
    L_path = ((L_lo-L_lo.mean())*(L_hi-L_hi.mean())+(L_lo.mean()*L_hi.mean()))**.5 
    

    # -- 4. final wheel ------------------------------------------------------
    Lab_final = np.stack((L_path, C0 * cθ, C0 * sθ), -1)
    wheel = XYZ_to_sRGB(Oklab_to_XYZ(Lab_final)).astype(np.float32)
    return wheel, C0, L_path,  L_lo, L_hi


import numpy as np
from colour import XYZ_to_sRGB, Oklab_to_XYZ

# ─────────────────────────────────────────────────────────────────────────────
#  Multi-harmonic constrained fitter  (fixed tuple-assignment bug)
# ─────────────────────────────────────────────────────────────────────────────
def _fit_sine_between(L_lo, L_hi, *,
                      order: int = 3,        # include sin(kθ), cos(kθ) for k=1…order
                      p: float = 1.5,        # weight exponent for narrow gaps
                      safety: float = 0.999, # final amplitude margin
                      tol: float = 1e-6,
                      max_iter: int = 50):
    """
    Fit  L(θ) = c + Σₖ Aₖ sin(kθ) + Bₖ cos(kθ)  inside  L_lo ≤ L ≤ L_hi.

    Returns L_fit that never crosses the envelopes.
    """
    N  = L_lo.size
    θ  = np.linspace(0.0, 2.0*np.pi, N, endpoint=False, dtype=np.float32)

    # — 1. weighted least-squares to corridor midpoint ————————————————
    mid = 0.5 * (L_lo + L_hi)
    w   = 1.0 / (L_hi - L_lo) ** p

    cols = [np.ones_like(θ)]
    for k in range(1, order + 1):
        cols.extend((np.sin(k*θ), np.cos(k*θ)))
    X = np.column_stack(cols)                         # (N, 2·order+1)

    β = np.linalg.lstsq(np.sqrt(w)[:, None] * X,
                        np.sqrt(w) * mid,
                        rcond=None)[0]
    c0, harm = β[0], X[:, 1:] @ β[1:]                # separate constant & harmonic

    # — 2. binary-search common scale factor s ————————————————
    s_lo, s_hi = 0.0, 1.0
    for _ in range(max_iter):
        s = 0.5 * (s_lo + s_hi)

        c_min = np.max(L_lo - s*harm)   # highest lower-bound
        c_max = np.min(L_hi - s*harm)   # lowest upper-bound

        if c_min <= c_max:              # feasible ⇒ try bigger amplitude
            s_lo = s
        else:                           # infeasible ⇒ shrink amplitude
            s_hi = s

        if s_hi - s_lo < tol:
            break

    s_opt  = safety * s_lo
    c_opt  = 0.5 * (np.max(L_lo - s_opt*harm) + np.min(L_hi - s_opt*harm))
    return c_opt + s_opt * harm


# ─────────────────────────────────────────────────────────────────────────────
#  Public wheel builder
# ─────────────────────────────────────────────────────────────────────────────
def build_slice_sine_opt_wheel(
        N: int   = 256,
        margin: float = 0.97,
        L_samples: int = 33,
        L_fine: int    = 301,
        iters: int     = 10,
        order: int     = 3,      # allow sin/cos up to k=order
        p: float       = 1.5,
        safety: float  = 0.999,
        tol: float     = 1e-6):
    """
    Largest-amplitude trigonometric series fitting between slice envelopes.
    """
    # --- envelopes & uniform chroma ----------------------------------------
    _, C0, _, L_lo, L_hi = build_slice_interpolated_wheel(
        N=N, margin=margin, L_samples=L_samples,
        L_fine=L_fine, iters=iters)

    # --- constrained multi-harmonic fit ------------------------------------
    L_fit = _fit_sine_between(L_lo, L_hi,
                              order=order, p=p,
                              safety=safety, tol=tol)

    # --- OK-Lab → sRGB wheel -----------------------------------------------
    θ  = np.linspace(0, 2*np.pi, N, endpoint=False, dtype=np.float32)
    cθ, sθ = np.cos(θ), np.sin(θ)
    Lab = np.stack((L_fit, C0*cθ, C0*sθ), axis=-1)
    wheel = XYZ_to_sRGB(Oklab_to_XYZ(Lab)).astype(np.float32)
    return wheel, C0, L_fit, L_lo, L_hi


# modified to use sinebow color
import colorsys
from cmap import Colormap  # isoluminant colormap support
def dx_to_circ(dP, transparency=False, mask=None,
               sinebow=1,
               iso=0, 
            #    iso_map="cmocean:phase",
            iso_map='oklch',
            offset = 0,
               norm=True):
    """ dP is 2 x Y x X => 'optic' flow representation 
    
    Parameters
    -------------
    
    dP: 2xLyxLx array
        Flow field components [dy,dx]
        
    transparency: bool, default False
        magnitude of flow controls opacity, not lightness (clear background)
        
    mask: 2D array 
        Multiplies each RGB component to suppress noise

    iso_map: str, default "cmocean:phase"
        When *iso* is ``True``, choose the cyclic constant-lightness palette.

        • ``"cmocean:phase"`` (default) - built-in cmocean isoluminant wheel.  
        • ``"oklch"`` - on-the-fly constant-L OKLCH circle with extra chroma.
    """
    
    dP = np.array(dP)
    mag = np.sqrt(np.sum(dP**2,axis=0))
    if norm:
        mag = np.clip(transforms.normalize99(mag,omni=OMNI_INSTALLED), 0, 1.)[...,np.newaxis]
    
    angles = np.arctan2(dP[1], dP[0]) + np.pi

    if iso:
        # if iso_map.lower() == "oklch":
        #     # angles -= 2*np.pi / 4      # rotate so red ≈ +X
        #     angles *= -1
        
        #     # ── constant-L OKLCH circle (L≈0.65, C≈0.19) ─────────────────
        #     t = (angles % (2 * np.pi)) / (2 * np.pi)        # 0-1
        #     # L0 = 0.65                                       # lightness
        #     # C0 = 0.19                                       # chroma safe in sRGB
            
        #     L0 = 0.75
        #     C0 = 0.12
            
        #     h  = t * 2 * np.pi
        #     a_ = C0 * np.cos(h)
        #     b_ = C0 * np.sin(h)

        #     # OKLab → linear RGB  (Ottosson 2020)
        #     l_ = (L0 + 0.3963377774 * a_ + 0.2158037573 * b_) ** 3
        #     m_ = (L0 - 0.1055613458 * a_ - 0.0638541728 * b_) ** 3
        #     s_ = (L0 - 0.0894841775 * a_ - 1.2914855480 * b_) ** 3

        #     r_lin =  4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_
        #     g_lin = -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_
        #     b_lin = -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_

        #     rgb_lin = np.stack([r_lin, g_lin, b_lin], axis=-1)

        #     # linear-RGB → sRGB (gamma 2.4)
        #     a = 0.055
        #     rgb = np.where(
        #         rgb_lin <= 0.0031308,
        #         12.92 * rgb_lin,
        #         (1 + a) * np.clip(rgb_lin, 0, 1) ** (1 / 2.4) - a
        #     )
        #     rgb = np.clip(rgb, 0, 1)
        
        if iso_map.lower() == "oklch":
            # Rotate and convert angle → phase ∈ [0,1]
            angles *= -1
            # angles -= np.pi/4
            angles+=offset
            t = (angles % (2 * np.pi)) / (2 * np.pi)

            # --- sample the pre-computed balanced OKLCH wheel --------------
            # 1024 samples give sub-pixel smoothness for images up to 4 k
            # wheel, _, _ = build_balanced_wheel(1024)       # (1024, 3) sRGB
            wheel, C0, L_fit, L_lo, L_hi = build_slice_sine_opt_wheel(order=2, p=1)     
            

            idx_float = t * wheel.shape[0]                 # fractional indices
            idx0      = np.floor(idx_float).astype(int) % wheel.shape[0]
            idx1      = (idx0 + 1) % wheel.shape[0]
            w1        = idx_float - idx0                   # interpolation weight

            # Linear interpolation between neighbouring wheel colours
            rgb = (1.0 - w1)[..., None] * wheel[idx0] + w1[..., None] * wheel[idx1]
        else:
            angles += np.pi / 4      # rotate so red ≈ +X
        
            cmap_iso = Colormap(iso_map)     # e.g. "cmocean:phase"
            t = (angles % (2 * np.pi)) / (2 * np.pi)
            rgb = np.reshape(
                np.array([cmap_iso(float(v)).rgba[:3] for v in t.ravel()]),
                t.shape + (3,)
            )

    elif sinebow:
        a = 2
        angles_shifted = np.stack(
            [angles, angles + 2 * np.pi / 3, angles + 4 * np.pi / 3],
            axis=-1
        )
        rgb = (np.cos(angles_shifted) + 1) / a
    else:
        r, g, b = colorsys.hsv_to_rgb(angles, 1, 1)
        rgb = np.stack((r, g, b), axis=0)
    if transparency:
        im = np.concatenate((rgb, mag), axis=-1)
    else:
        im = rgb * mag

    if mask is not None and transparency and dP.shape[0] < 3:
        im[:, :, -1] *= mask

    im = (np.clip(im, 0, 1) * 255).astype(np.uint8)
    return im

from omnipose.plot import imshow
def show_segmentation(fig, img, maski, flowi, bdi=None, channels=None, file_name=None, omni=False, 
                      seg_norm=False, bg_color=None, outline_color=[1,0,0], img_colors=None,
                      channel_axis=-1, 
                      figsize=(12, 3), dpi=300,  # figsize and dpi for matplotlib figure
                      hold=False, 
                      interpolation='bilinear'):
    """ plot segmentation results (like on website)
    
    Can save each panel of figure with file_name option. Use channels option if
    img input is not an RGB image with 3 channels.
    
    Parameters
    -------------

    fig: matplotlib.pyplot.figure
        figure in which to make plot

    img: 2D or 3D array
        image input into cellpose

    maski: int, 2D array
        for image k, masks[k] output from cellpose_omni.eval, where 0=NO masks; 1,2,...=mask labels

    flowi: int, 2D array 
        for image k, flows[k][0] output from cellpose_omni.eval (RGB of flows)

    channels: list of int (optional, default [0,0])
        channels used to run Cellpose, no need to use if image is RGB

    file_name: str (optional, default None)
        file name of image, if file_name is not None, figure panels are saved
        
    omni: bool (optional, default False)
        use omni version of normalize99, image_to_rgb
        
    seg_norm: bool (optional, default False)
        improve cell visibility under labels
        
    bg_color: float (Optional, default none)
        background color to draw behind flow (visible if flow transparency is on)
        
    img_colors: NDarray, float (Optional, default none)
        colors to which each image channel will be mapped (multichannel defaults to sinebow)
        

    """
    
    if fig is None:
        fig, ax = figure(figsize=figsize, dpi=dpi)

    if channels is None:
        channels = [0,0]
    img0 = img.copy()
    
    if img0.ndim==2:
        # this catches grayscale, converts to standard RGB YXC format 
        img0 = image_to_rgb(img0, channels=channels, omni=omni)
    else:
        # otherwise it must actually have some channels
        
        # no channel axis specified means we shoudl assume it is CYX format
        if channel_axis is None:
            channel_axis = 0 

        # this branch catches cases where RGB image is CYX, converts to YXC
        if img0.shape[0]==3 and channel_axis!=-1:
            img0 = np.transpose(img0, (1,2,0))

        # for anything else
        if img0.shape[channel_axis]!=3:
            # need to convert the image to CYX first 
            img0 = transforms.move_axis_new(img0,channel_axis,0)
            img0 = colorize(img0,colors=img_colors)
    
    img0 = (transforms.normalize99(img0,omni=omni)*(2**8-1)).astype(np.uint8)
    

    if bdi is None or not bdi.shape:
        outlines = utils.masks_to_outlines(maski,omni)
    else:
        outlines = bdi

    # Image normalization to improve cell visibility under labels
    if seg_norm:
        fg = 1/9
        p = np.clip(transforms.normalize99(img0,omni=omni), 0, 1)
        img1 = p**(np.log(fg)/np.log(np.mean(p[maski>0])))
    else:
        img1 = img0
    
    # the mask_overlay function changes colors (preserves only hue I think). The label2rgb function from
    # skimage.color works really well. 
    if omni and SKIMAGE_ENABLED and OMNI_INSTALLED:
        m,n = ncolor.label(maski,max_depth=20,return_n=True) if np.any(maski) else (maski,1)
        # print(m.shape, n)
        c = sinebow(n)
        clrs = np.array(list(c.values()))[1:]
        
        # print('colors',clrs)
        img1 = rescale(color.rgb2gray(img1))
        overlay = color.label2rgb(m,img1,clrs,
                                  bg_label=0,
                                  alpha=np.stack([((m>0)*1.+outlines*0.75)/3]*3,axis=-1))
    

    else:
        overlay = mask_overlay(img0, maski)

        
    outli = outline_view(img0, maski, boundaries=outlines, color=np.array(outline_color)*255,
                        channels=channels, channel_axis=channel_axis, skip_formatting=True)

    ax = fig.get_axes()[0]
    fig = imshow([img0, outli, overlay, flowi], ax=ax, 
                titles=['original image',
                        'predicted outlines',
                        'predicted masks',
                        'predicted flow field'], 
            interpolation=interpolation, hold=hold, 
            figsize=figsize, dpi=dpi)

    
    if file_name is not None:
        save_path = os.path.splitext(file_name)[0]
        io.imsave(save_path + '_overlay.jpg', overlay)
        io.imsave(save_path + '_outlines.jpg', outli)
        io.imsave(save_path + '_flows.jpg', flowi)
        
            
    if hold:
        return fig, img1, outlines, overlay 
        

def mask_rgb(masks, colors=None):
    """ masks in random rgb colors

    Parameters
    ----------------

    masks: int, 2D array
        masks where 0=NO masks; 1,2,...=mask labels

    colors: int, 2D array (optional, default None)
        size [nmasks x 3], each entry is a color in 0-255 range

    Returns
    ----------------

    RGB: uint8, 3D array
        array of masks overlaid on grayscale image

    """
    if colors is not None:
        if colors.max()>1:
            colors = np.float32(colors)
            colors /= 255
        colors = utils.rgb_to_hsv(colors)
    
    HSV = np.zeros((masks.shape[0], masks.shape[1], 3), np.float32)
    HSV[:,:,2] = 1.0
    for n in range(int(masks.max())):
        ipix = (masks==n+1).nonzero()
        if colors is None:
            HSV[ipix[0],ipix[1],0] = np.random.rand()
        else:
            HSV[ipix[0],ipix[1],0] = colors[n,0]
        HSV[ipix[0],ipix[1],1] = np.random.rand()*0.5+0.5
        HSV[ipix[0],ipix[1],2] = np.random.rand()*0.5+0.5
    RGB = (utils.hsv_to_rgb(HSV) * 255).astype(np.uint8)
    return RGB

def mask_overlay(img, masks, colors=None, omni=False):
    """ overlay masks on image (set image to grayscale)

    Parameters
    ----------------

    img: int or float, 2D or 3D array
        img is of size [Ly x Lx (x nchan)]

    masks: int, 2D array
        masks where 0=NO masks; 1,2,...=mask labels

    colors: int, 2D array (optional, default None)
        size [nmasks x 3], each entry is a color in 0-255 range

    Returns
    ----------------

    RGB: uint8, 3D array
        array of masks overlaid on grayscale image

    """
    if colors is not None:
        if colors.max()>1:
            colors = np.float32(colors)
            colors /= 255
        colors = utils.rgb_to_hsv(colors)
    if img.ndim>2:
        img = img.astype(np.float32).mean(axis=-1)
    else:
        img = img.astype(np.float32)
    
    HSV = np.zeros((img.shape[0], img.shape[1], 3), np.float32)
    HSV[:,:,2] = np.clip((img / 255. if img.max() > 1 else img) * 1.5, 0, 1)
    hues = np.linspace(0, 1, masks.max()+1)[np.random.permutation(masks.max())]
    for n in range(int(masks.max())):
        ipix = (masks==n+1).nonzero()
        if colors is None:
            HSV[ipix[0],ipix[1],0] = hues[n]
        else:
            HSV[ipix[0],ipix[1],0] = colors[n,0]
        HSV[ipix[0],ipix[1],1] = 1.0
    RGB = (utils.hsv_to_rgb(HSV) * 255).astype(np.uint8)
    return RGB

def image_to_rgb(img0, channels=None, channel_axis=-1, omni=False):
    """ image is 2 x Ly x Lx or Ly x Lx x 2 - change to RGB Ly x Lx x 3 """

    
    img = img0.copy()
    img = img.astype(np.float32)
    if img.ndim<3: # if monochannel 
        img = img[...,np.newaxis]
        channels = [0,0]
    if img.shape[0]<5: # putting channel axis last 
        img = np.transpose(img, (1,2,0))
        
    # if channels is still none, ndim>2
    if channels is None:
        if np.all(img0[...,0]==img0[...,1]):
            channels = [0,0] # if R=G, assume grayscale image 
        else:
            channels = [i+1 for i in range(img0.shape[channel_axis])] # 1,2,3 for axes 0,1,2
        
    if channels[0]==0:
        img = img.mean(axis=-1)[:,:,np.newaxis]
    for i in range(img.shape[-1]):
        if np.ptp(img[:,:,i])>0:
            img[:,:,i] = transforms.normalize99(img[:,:,i],omni=omni)
            img[:,:,i] = np.clip(img[:,:,i], 0, 1) #irrelevant for omni
    
    img *= 255
    img = np.uint8(img)
    RGB = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    
    # at this point, channel axis is last
    if img.shape[-1]==1:
        RGB = np.tile(img,(1,1,3))
    else:
        RGB[:,:,channels[0]-1] = img[:,:,0]
        if channels[1] > 0:
            RGB[:,:,channels[1]-1] = img[:,:,1]
    return RGB

def interesting_patch(mask, bsize=130):
    """ get patch of size bsize x bsize with most masks """
    Ly,Lx = mask.shape
    m = np.float32(mask>0)
    m = gaussian_filter(m, bsize/2)
    y,x = np.unravel_index(np.argmax(m), m.shape)
    ycent = max(bsize//2, min(y, Ly-bsize//2))
    xcent = max(bsize//2, min(x, Lx-bsize//2))
    patch = [np.arange(ycent-bsize//2, ycent+bsize//2, 1, int),
             np.arange(xcent-bsize//2, xcent+bsize//2, 1, int)]
    return patch

def disk(med, r, Ly, Lx):
    """ returns pixels of disk with radius r and center med """
    yy, xx = np.meshgrid(np.arange(0,Ly,1,int), np.arange(0,Lx,1,int),
                         indexing='ij')
    inds = ((yy-med[0])**2 + (xx-med[1])**2)**0.5 <= r
    y = yy[inds].flatten()
    x = xx[inds].flatten()
    return y,x

def outline_view(img0, maski, boundaries=None, color=[1,0,0], 
                 channels=None, channel_axis=-1, 
                 mode='inner',connectivity=2, skip_formatting=False):
    """
    Generates a red outline overlay onto image.
    """
#     img0 = utils.rescale(img0)
    if np.max(color)<=1 and not skip_formatting:
        color = np.array(color)*(2**8-1)
    
    if not skip_formatting:
        img0 = image_to_rgb(img0, channels=channels, channel_axis=channel_axis, omni=True)
    
    if boundaries is None:
        if SKIMAGE_ENABLED:
            outlines = find_boundaries(maski,mode=mode,connectivity=connectivity) #not using masks_to_outlines as that gives border 'outlines'
        else:
            outlines = utils.masks_to_outlines(maski,mode=mode) 
            
    else:
        outlines = boundaries
    outY, outX = np.nonzero(outlines)
    imgout = img0.copy()
    
    imgout[outY, outX] = color
    return imgout
