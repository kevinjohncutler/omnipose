import numpy as np
import torch
from scipy.ndimage import convolve1d, convolve, gaussian_filter



def moving_average(x, w):
    return convolve1d(x,np.ones(w)/w,axis=0)


def curve_filter(im,filterWidth=1.5):
    """
    curveFilter : calculates the curvatures of an image.

    INPUT
    _____
    im : image to be filtered
    filterWidth : filter width
           
    OUTPUT
    ------
    ``M_`` : mean curvature (negatives zeroed)  
    ``G_`` : Gaussian curvature (negatives zeroed)  
    ``C1_``: principal curvature 1 (negatives zeroed)  
    ``C2_``: principal curvature 2 (negatives zeroed)  
    ``M``  : mean curvature  
    ``G``  : Gaussian curvature  
    ``C1`` : principal curvature 1  
    ``C2`` : principal curvature 2  
    ``im_xx`` : ∂²x / ∂x²  
    ``im_yy`` : ∂²x / ∂y²  
    ``im_xy`` : ∂²x / ∂x∂y

    """
    shape = [np.floor(7*filterWidth) //2 *2 +1]*2 # minor modification is to make this odd
    
    m,n = [(s-1.)/2. for s in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    v = filterWidth**2
    gau = 1/(2*np.pi*v) * np.exp( -(x**2 + y**2) / (2.*v) )
    
    
    f_xx = ((x/v)**2-1/v)*gau
    f_yy = ((y/v)**2-1/v)*gau
    f_xy = y*x*gau/v**2
    
    im_xx = convolve(im, f_xx, mode='nearest')
    im_yy = convolve(im, f_yy, mode='nearest')
    im_xy = convolve(im, f_xy, mode='nearest')
    
    # gaussian curvature
    G = im_xx*im_yy-im_xy**2

    # mean curvature
    M = -(im_xx+im_yy)/2

    # compute principal curvatures
    C1 = (M-np.sqrt(np.abs(M**2-G)));
    C2 = (M+np.sqrt(np.abs(M**2-G)));

    
    # remove negative values
    G_ = G.copy()
    G_[G<0] = 0;

    M_ = M.copy()
    M_[M<0] = 0

    C1_ = C1.copy()
    C1_[C1<0] = 0

    C2_ = C2.copy()
    C2_[C2<0] = 0

    return M_, G_, C1_, C2_, M, G, C1, C2, im_xx, im_yy, im_xy




def add_poisson_noise(image):
    noisy_image = np.random.poisson(image)
    noisy_image = np.clip(noisy_image, 0, 1)  # Clip values to [0, 1] range
    return noisy_image



def hysteresis_threshold(image, low, high):
    """
    Pytorch implementation of skimage.filters.apply_hysteresis_threshold(). 
    Discprepencies occur for very high thresholds/thin objects. 
    
    """
    # Ensure the image is a torch tensor
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image)

    # Create masks for values greater than low and high thresholds
    mask_low = image > low
    mask_high = image > high

    # Initialize thresholded tensor
    thresholded = mask_low.clone()

    # Create hysteresis kernel
    spatial_dims = len(image.shape) - 2
    kernel_size = [3] * spatial_dims
    hysteresis_kernel = torch.ones([1, 1] + kernel_size, device=image.device, dtype=image.dtype)

    # Hysteresis thresholding
    thresholded_old = torch.zeros_like(thresholded)
    while (thresholded_old != thresholded).any():
        if spatial_dims == 2:
            hysteresis_magnitude = torch.nn.functional.conv2d(thresholded.float(), hysteresis_kernel, padding=1)
        elif spatial_dims == 3:
            hysteresis_magnitude = torch.nn.functional.conv3d(thresholded.float(), hysteresis_kernel, padding=1)
        else:
            raise ValueError(f'Unsupported number of spatial dimensions: {spatial_dims}')

        # thresholded_old = thresholded.clone()
        thresholded_old.copy_(thresholded)
        thresholded = ((hysteresis_magnitude > 0) & mask_low) | mask_high


    # sum_old = thresholded.sum()
    # while True:
    #     if spatial_dims == 2:
    #         hysteresis_magnitude = F.conv2d(thresholded.float(), hysteresis_kernel, padding=1)
    #     elif spatial_dims == 3:
    #         hysteresis_magnitude = F.conv3d(thresholded.float(), hysteresis_kernel, padding=1)
    #     else:
    #         raise ValueError(f'Unsupported number of spatial dimensions: {spatial_dims}')

    #     thresholded = ((hysteresis_magnitude > 0) & mask_low) | mask_high
    #     sum_new = thresholded.sum()

    #     if sum_new == sum_old:
    #         break

        # sum_old = sum_new
    return thresholded.bool()#, mask_low, mask_high


def correct_illumination(img,sigma=5):
    # Apply a Gaussian blur to the image
    blurred = gaussian_filter(img, sigma=sigma)

    # Normalize the image
    return (img - blurred) / np.std(blurred)
