from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift as im_shift
from scipy.ndimage import gaussian_filter

import fastremap 
import numpy as np
from .gpu import torch_GPU, torch_CPU
import dask


def shifts_to_slice(shifts, shape, pad=0):
    """
    Find the minimal crop box from time lapse registration shifts.
    """    
    # Convert shifts to integers
    shifts = np.round(shifts).astype(int)
    
    # Create a slice for each dimension
    slices = tuple(slice(max(0, np.max(shifts[:, dim])-pad), 
                         min(shape[dim], shape[dim] + np.min(shifts[:, dim])+pad))
                   for dim in range(shifts.shape[1]))
    
    return slices
    

def make_unique(masks):
    """Relabel stack of label matrices such that there is no repeated label across slices."""
    masks = masks.copy().astype(np.uint32)
    T = range(len(masks))
    offset = 0 
    for t in T:
        # f = format_labels(masks[t],clean=True)
        fastremap.renumber(masks[t],in_place=True)
        masks[t][masks[t]>0]+=offset
        offset = masks[t].max()
    return masks
    

def normalize_stack(vol,mask,bg=0.5,bright_foreground=None,
                    subtractive=False,iterations=1,equalize_foreground=1,quantiles=[0.01,0.99]):
    """
    Adjust image stacks so that background is 
    (1) consistent in brightness and 
    (2) brought to an even average via semantic gamma normalization.
    """
    # vol = rescale(vol)
    vol = vol.copy()
    # binarize background mask, recede from foreground, slice-wise to not erode in time
    kwargs = {'iterations':iterations} if iterations>1 else {}
    bg_mask = [binary_erosion(m==0,**kwargs) for m in mask] 
    # find mean backgroud for each slice
    bg_real = [np.nanmean(v[m]) for v,m in zip(vol,bg_mask)] 
    
    # automatically determine if foreground objects are bright or dark 
    if bright_foreground is None:
        bright_foreground = np.mean(vol[bg_mask]) < np.mean(vol[mask>0])
    
    # if smooth: 
    #     bg_real = moving_average(bg_real,5) 
    # some weird fluctuations happening with background being close to zero, but just on fluoresnece... might need to invert or go by foreground
    
    bg_min = np.min(bg_real) # get the minimum one, want to normalize by lowest one 
    
    # normalize wrt background
    if subtractive:
        vol = np.stack([safe_divide(v-bg_r,bg_min) for v,bg_r in zip(vol,bg_real)]) 
    else:
        vol = np.stack([v*safe_divide(bg_min,bg_r) for v,bg_r in zip(vol,bg_real)]) 
    # print('mm',vol.min(),vol.max(),bright_foreground)
    # equalize foreground signal
    if equalize_foreground:
        q1,q2 = quantiles
    
        if bright_foreground:
            fg_real = [np.percentile(v[m>0],99.99) for v,m in zip(vol,mask)]
            # fg_real = [v.max() for v,m in zip(vol,bg_mask)]    
            floor = np.percentile(vol[bg_mask],0.01)
            vol = [rescale(v,ceiling=f, floor=floor) for v,f in zip(vol,fg_real)]
        else:
            fg_real = [np.quantile(v[m>0],q1) for v,m in zip(vol,mask)]
            # fg_real = [.5]*(len(vol))
            # ceiling = np.percentile(vol[bg_mask],99.99)
            
            # print('hh',np.any(np.stack(fg_real)<0),np.any(np.stack(fg_real)>ceiling),ceiling,np.mean(fg_real))
            # vol = [rescale(v,ceiling=ceiling,floor=f) for v,f in zip(vol,fg_real)]
            # ceiling =  [np.percentile(v[m],99.99) for v,m in zip(vol,mask==0)]#bg_mask
            ceiling =  np.quantile(vol,q2,axis=(-2,-1))
            vol = [np.interp(v,(f, c), (0, 1)) for v,f,c in zip(vol,fg_real,ceiling)]
            
    # print([(np.max(v),np.min(v)) for v,bg_m in zip(vol,bg_mask)])
    vol = np.stack(vol)
    
    # vol = rescale(vol) # now rescale by overall min and max 
    vol = np.stack([v**(np.log(bg)/np.log(np.mean(v[bg_m]))) for v,bg_m in zip(vol,bg_mask)]) # now can gamma normalize 
    return vol

    
# import imreg_dft 
def cross_reg(imstack,upsample_factor=100,
              normalization=None,
              reverse=False, 
              localnorm=True, 
              max_shift=50, 
              order=1,
              moving_reference=False, ):
    """
    Find the transformation matrices for all images in a time series to align to the beginning frame. 
    """
    dim = imstack.ndim - 1 # dim is spatial, assume first dimension is t
    s = np.zeros(dim)
    shape = imstack.shape[-dim:]
    
    images_to_register = imstack if not reverse else imstack[::-1]

    # Now images_to_register[i] is the sum of image_stack over the interval slices[i]
    if localnorm: 
        images_to_register = images_to_register/gaussian_filter(images_to_register,sigma=[0,1,1])
    
    
    if moving_reference:
        shift_vectors = [[]]*len(images_to_register)
    
        for i,im in enumerate(images_to_register):
            if i==0:
                ref = im
                shift_vectors[i] = np.zeros(dim)
            else:
                shift = phase_cross_correlation(ref, 
                                                im, 
                                                upsample_factor = upsample_factor,
                                                normalization=normalization)[0] 
    
                if np.linalg.norm(shift) > max_shift:
                    shift = np.zeros_like(shift)
                    
                shift_vectors[i] = shift
                ref = im_shift(im,shift,cval=np.mean(im),order=order)

    
    else:  
        shift_vectors = [phase_cross_correlation(images_to_register[i], 
                                                images_to_register[i+1], 
                                                upsample_factor = upsample_factor,
                                                normalization=normalization)[0] for i in range(len(images_to_register)-1)]
        
    if not moving_reference:
        shift_vectors.insert(0, np.asarray([0.0,0.0]))  
        
    shift_vectors = np.stack(shift_vectors)
    
    if reverse:
        shift_vectors = shift_vectors[::-1] * (-1 if not moving_reference else 1)


    if not moving_reference:

        shift_vectors = np.where(np.linalg.norm(shift_vectors,axis=1, keepdims=1) > max_shift, 0, shift_vectors)
        shift_vectors = np.cumsum(shift_vectors, axis=0)
        
    
    shift_vectors -= np.mean(shift_vectors,axis=0)
    

    return shift_vectors 

def shift_stack(imstack, shift_vectors, order=1, cval=None, prefilter=True, mode='nearest'):
    """
    Shift each time slice of imstack according to list of nD shifts. 
    """
    imstack = imstack.astype(np.float32)
    shift_vectors = shift_vectors.astype(np.float32)

    # delayed_images = da.from_array(imstack).to_delayed()
    
    ndim = imstack.ndim
    axes = tuple(range(-(ndim-1),0))
    cvals = np.nanmean(imstack,axis=axes) if cval is None else [cval]*len(shift_vectors)
    mode = mode if cval is None else 'constant'
    
    # Apply the shift to each image
    shifted_images = [dask.delayed(im_shift)(image,
                                            shift_vector, 
                                            order=order,
                                            prefilter=prefilter,
                                            mode=mode,
                                            cval=cv,
                                            ) for image, shift_vector, cv in zip(imstack, shift_vectors, cvals)]

    # Compute the shifted images in parallel
    shifted_images = dask.compute(*shifted_images)
    shifted_images = np.stack(shifted_images, axis=0)
    
    return shifted_images

# GPU version
# import torch
# import torch.fft

# def phase_cross_correlation_GPU(target, moving_images):

#     # Assuming target is a 2D tensor [height, width]
#     # and moving_images is a 3D tensor [num_images, height, width]

#     # Expand dims of target to match moving_images
#     target = target.unsqueeze(0)
#     # print(target.shape,moving_images.shape)
#     # Compute FFT of images
#     target_fft = torch.fft.fftn(target, dim=[-2, -1])
#     moving_fft = torch.fft.fftn(moving_images, dim=[-2, -1])
    
#     # print(target_fft.shape,moving_fft.shape)
    
#     # Compute cross-correlation by multiplying with complex conjugate
#     cross_corr = torch.fft.ifftn(target_fft * moving_fft.conj(), dim=[-2, -1]).real
    
#     # Find peak in cross-correlation
#     max_indices = torch.argmax(cross_corr.view(cross_corr.shape[0], -1), dim=1)
    
#     # Convert flat indices to 2D indices
#     height = cross_corr.shape[-2]
#     width = cross_corr.shape[-1]
#     shifts_y = max_indices // width
#     shifts_x = max_indices % width

#     # Adjust shifts to fall within the correct range
#     # make sure shift vector points in the right direction 
#     shifts_y =  height // 2 - (shifts_y + height // 2) % height
#     shifts_x =  width // 2 - (shifts_x + width // 2) % width

#     # Combine shifts along both dimensions into a single tensor
#     shifts = torch.stack([shifts_y, shifts_x], dim=-1)
#     return shifts

# def phase_cross_correlation_GPU(image_stack, target_index):
#     # Assuming image_stack is a 3D tensor [num_images, height, width]
#     # and target_index is an integer

#     target_image = image_stack[target_index].unsqueeze(0)
#     moving_images = torch.cat([image_stack[:target_index], image_stack[target_index+1:]])

#     target_fft = torch.fft.fftn(target_image, dim=[-2, -1])
#     moving_fft = torch.fft.fftn(moving_images, dim=[-2, -1])

#     cross_corr = torch.fft.ifftn(target_fft * moving_fft.conj(), dim=[-2, -1]).real

#     max_indices = torch.argmax(cross_corr.view(cross_corr.shape[0], -1), dim=1)

#     height = cross_corr.shape[-2]
#     width = cross_corr.shape[-1]
#     shifts_y = max_indices // width
#     shifts_x = max_indices % width

#     shifts_y =  height // 2 - (shifts_y + height // 2) % height
#     shifts_x =  width // 2 - (shifts_x + width // 2) % width

#     shifts = torch.stack([shifts_y, shifts_x], dim=-1)

#     # Insert a zero shift at the target index
#     zero_shift = torch.zeros(1, 2, device=image_stack.device)
#     shifts = torch.cat([shifts[:target_index], zero_shift, shifts[target_index:]])

#     return shifts.long()
    


# import torch.nn.functional as F

# def phase_cross_correlation_GPU(image_stack, target_index, upsample_factor=1):
#     # Assuming image_stack is a 3D tensor [num_images, height, width]
#     # and target_index is an integer

#     target_image = image_stack[target_index].unsqueeze(0)
#     moving_images = torch.cat([image_stack[:target_index], image_stack[target_index+1:]])

#     target_fft = torch.fft.fftn(target_image, dim=[-2, -1])
#     moving_fft = torch.fft.fftn(moving_images, dim=[-2, -1])

#     cross_corr = torch.fft.ifftn(target_fft * moving_fft.conj(), dim=[-2, -1]).real

#     # Upsample cross correlation to achieve subpixel precision
#     if upsample_factor > 1:
#         cross_corr = cross_corr.unsqueeze(1)
#         print('cc',cross_corr.shape)
#         cross_corr = F.interpolate(cross_corr, scale_factor=upsample_factor, 
#                                    mode='bilinear', align_corners=False)
        
#         print('cc',cross_corr.shape)
        
#         cross_corr = cross_corr.squeeze(1)

#     max_indices = torch.argmax(cross_corr.view(cross_corr.shape[0], -1), dim=1)

#     height = cross_corr.shape[-2]
#     width = cross_corr.shape[-1]
#     shifts_y = max_indices // width
#     shifts_x = max_indices % width
#     shifts_y =  height // 2 - (shifts_y + height // 2) % height
#     shifts_x =  width // 2 - (shifts_x + width // 2) % width

#     # Convert shifts back to original pixel grid
#     shifts_y = shifts_y / upsample_factor
#     shifts_x = shifts_x / upsample_factor

#     shifts = torch.stack([shifts_y, shifts_x], dim=-1)

#     # Insert a zero shift at the target index
#     zero_shift = torch.zeros(1, 2, device=image_stack.device)
#     shifts = torch.cat([shifts[:target_index], zero_shift, shifts[target_index:]])
#     return shifts

# def phase_cross_correlation_GPU(image_stack, target_index, upsample_factor=10):
#     # Assuming image_stack is a 3D tensor [num_images, height, width]
#     # and target_index is an integer

#     # Upsample the images
#     image_stack = F.interpolate(image_stack.unsqueeze(1).float(), scale_factor=upsample_factor, mode='bilinear', align_corners=False).squeeze(1)

#     target_image = image_stack[target_index].unsqueeze(0)
#     moving_images = torch.cat([image_stack[:target_index], image_stack[target_index+1:]])

#     target_fft = torch.fft.fftn(target_image, dim=[-2, -1])
#     moving_fft = torch.fft.fftn(moving_images, dim=[-2, -1])

#     cross_corr = torch.fft.ifftn(target_fft * moving_fft.conj(), dim=[-2, -1]).real

#     max_indices = torch.argmax(cross_corr.view(cross_corr.shape[0], -1), dim=1)

#     height = cross_corr.shape[-2]
#     width = cross_corr.shape[-1]
#     shifts_y = max_indices // width
#     shifts_x = max_indices % width

#     shifts_y =  height // 2 - (shifts_y + height // 2) % height
#     shifts_x =  width // 2 - (shifts_x + width // 2) % width

#     # Convert shifts back to original pixel grid
#     shifts_y = shifts_y / upsample_factor
#     shifts_x = shifts_x / upsample_factor

#     shifts = torch.stack([shifts_y, shifts_x], dim=-1)

#     # Insert a zero shift at the target index
#     zero_shift = torch.zeros(1, 2, device=image_stack.device)
#     shifts = torch.cat([shifts[:target_index], zero_shift, shifts[target_index:]])

#     return shifts.float()



def gaussian_kernel(size: int, sigma: float, device=torch_GPU):
    """Creates a 2D Gaussian kernel with mean 0.

    Args:
        size (int): The size of the kernel. Should be an odd number.
        sigma (float): The standard deviation of the Gaussian distribution.

    Returns:
        torch.Tensor: The Gaussian kernel.
    """
    coords = torch.arange(size,device=device).float() - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    return g.outer(g)

def apply_gaussian_blur(image, kernel_size, sigma, device=torch_GPU):
    """Applies a Gaussian blur to the image.

    Args:
        image (torch.Tensor): The image to blur.
        kernel_size (int): The size of the Gaussian kernel.
        sigma (float): The standard deviation of the Gaussian distribution.

    Returns:
        torch.Tensor: The blurred image.
    """
    kernel = gaussian_kernel(kernel_size, sigma, device).unsqueeze(0).unsqueeze(0)
    image = image.unsqueeze(0).unsqueeze(0)

    # Apply 'reflect' padding to the image
    padding_size = kernel_size // 2
    image = F.pad(image, (padding_size, padding_size, padding_size, padding_size), mode='reflect')

    # Perform the convolution without additional padding
    blurred = F.conv2d(image, kernel, padding=0)

    return blurred.squeeze(0).squeeze(0)

def phase_cross_correlation_GPU_old(image_stack, target_index=None, upsample_factor=10, 
                                reverse=False,normalize=False):
    # Assuming image_stack is a 3D tensor [num_images, height, width]
    # and target_index is an integer or None for sequential registration

    
    im_to_reg = torch.stack([i/apply_gaussian_blur(i, 5, 1) for i in image_stack])

    # Upsample the images
    image_stack = F.interpolate(im_to_reg.unsqueeze(1).float(), 
                                scale_factor=upsample_factor, mode='bilinear', 
                                align_corners=False).squeeze(1)

    # Initialize shifts with a zero shift for the first image
    # shifts = [[0, 0]]
    shifts = []
    
    for i in range(1, len(image_stack)):
        if target_index is None:
            # Sequential registration
            # target_image = image_stack[i-1]
            if reverse:
                # Reverse registration
                target_image = image_stack[i+1] if i < len(image_stack) - 1 else image_stack[i]
            else:
                # Sequential registration
                target_image = image_stack[i-1] if i > 0 else image_stack[i]
        else:
            # Target registration
            target_image = image_stack[target_index]

        moving_image = image_stack[i]

        # target_fft = torch.fft.fftn(target_image.unsqueeze(0), dim=[-2, -1])
        # moving_fft = torch.fft.fftn(moving_image.unsqueeze(0), dim=[-2, -1])
        target_fft = torch.fft.fftn(target_image, dim=[-2, -1])
        moving_fft = torch.fft.fftn(moving_image, dim=[-2, -1])

        # Compute the cross-power spectrum
        cross_power_spectrum = target_fft * moving_fft.conj()

        # Normalize the cross-power spectrum if the normalize option is True
        if normalize:
            cross_power_spectrum /= torch.abs(cross_power_spectrum)
        
        cross_corr = torch.abs(torch.fft.ifftn(cross_power_spectrum, dim=[-2, -1]))
        print('cc',cross_corr.shape)
        max_index = torch.argmax(cross_corr.view(-1))

        height = cross_corr.shape[-2]
        width = cross_corr.shape[-1]
        shift_y = max_index // width
        shift_x = max_index % width

        shift_y =  height // 2 - (shift_y + height // 2) % height
        shift_x =  width // 2 - (shift_x + width // 2) % width

        # Convert shifts back to original pixel grid
        shift_y = shift_y / upsample_factor
        shift_x = shift_x / upsample_factor

        shifts.append([shift_y, shift_x])

    shifts.append([0,0])
    shifts = torch.tensor(shifts, device=image_stack.device)*(-2)

    # Subtract the average shift from all shifts to minimize the total shift
    # avg_shift = shifts.mean(dim=0)
    # shifts -= avg_shift
    # shifts = torch.cumsum(shifts,dim=0)
    
    return shifts.float()
    
#     return accumulated_shifts
def phase_cross_correlation_GPU(image_stack, 
                                upsample_factor=10, 
                                # normalization='phase'
                                normalization=None,
                                
                                ):
    # Assuming image_stack is a 3D tensor [num_images, height, width]
    
    # Upsample the images
    # image_stack = F.interpolate(image_stack.unsqueeze(1).float(), 
    #                             scale_factor=upsample_factor, mode='bilinear', 
    #                             align_corners=False).squeeze(1)
    
    # m = torch.nn.Upsample(scale_factor=tuple([upsample_factor,upsample_factor]),mode='bilinear')
    # image_stack = m(image_stack.float().unsqueeze(1)).squeeze(1)
    device = image_stack.device
    
    im_to_reg = torch.stack([i/apply_gaussian_blur(i, 9, 3, device=device) for i in image_stack.float()])
    # im_to_reg = image_stack
    # Compute the FFT of the images
    norm='backward'
    image_fft = torch.fft.fft2(im_to_reg,norm=norm)#, dim=[-2, -1])
    
    # Compute the cross-power spectrum for each pair of images
    cross_power_spectrum = image_fft[:-1] * image_fft[1:].conj()
    
    # Normalize the cross-power spectrum
    if normalization == 'phase':
        cross_power_spectrum /= torch.abs(cross_power_spectrum)#+1e-6
    
    # Compute the cross-correlation by taking the inverse FFT
    cross_corr = torch.abs(torch.fft.ifft2(cross_power_spectrum,norm=norm)) #, dim=[-2, -1])
    m = torch.nn.Upsample(scale_factor=upsample_factor,mode='bilinear')
    cross_corr = m(cross_corr.unsqueeze(1)).squeeze(1)
    
    # Find the shift for each pair of images
    max_indices = torch.argmax(cross_corr.view(cross_corr.shape[0], -1), dim=-1).float()
    shifts_y, shifts_x = (max_indices / cross_corr.shape[-1]).long(), (max_indices % cross_corr.shape[-1]).long()

    # Stack the shifts and append a [0, 0] shift at the beginning
    # shifts = torch.stack([shifts_y, shifts_x]).T
    shifts = 2*torch.stack([shifts_y, shifts_x]).T
    zero_shift = torch.zeros(1, 2, dtype=shifts.dtype, device=shifts.device)
    shifts = torch.cat([shifts,zero_shift], dim=0) / upsample_factor

    # Accumulate the shifts - SUPER important and was the cause of the bug 
    shifts = torch.cumsum(shifts.flip(dims=[0]),dim=0).flip(dims=[0])
    
    # Subtract the average shift from all shifts to minimize the total shift
    avg_shift = shifts.mean(dim=0)
    shifts -= avg_shift

    # should replace shift by making it so that the shifts are closest to pixel shifts? 

    return shifts

# ### below two functions an experiment 
# def pairwise_registration(image_stack, upsample_factor=10):

#     im_to_reg = torch.stack([i/apply_gaussian_blur(i, 5, 5) for i in image_stack])

#     # Upsample the images
#     image_stack = F.interpolate(im_to_reg.unsqueeze(1).float(), scale_factor=upsample_factor, mode='bilinear', align_corners=False).squeeze(1)

#     num_images = len(image_stack)
#     shifts = torch.zeros((num_images, num_images, 2), device=image_stack.device)

#     for i in range(num_images):
#         for j in range(i+1, num_images):
#             target_image = image_stack[i]
#             moving_image = image_stack[j]

#             target_fft = torch.fft.fftn(target_image.unsqueeze(0), dim=[-2, -1])
#             moving_fft = torch.fft.fftn(moving_image.unsqueeze(0), dim=[-2, -1])

#             cross_corr = torch.fft.ifftn(target_fft * moving_fft.conj(), dim=[-2, -1]).real

#             max_index = torch.argmax(cross_corr.view(-1))

#             height = cross_corr.shape[-2]
#             width = cross_corr.shape[-1]
#             shift_y = max_index // width
#             shift_x = max_index % width

#             shift_y =  height // 2 - (shift_y + height // 2) % height
#             shift_x =  width // 2 - (shift_x + width // 2) % width

#             # Convert shifts back to original pixel grid
#             shift_y = shift_y / upsample_factor
#             shift_x = shift_x / upsample_factor

#             shifts[i, j] = torch.tensor([shift_y, shift_x])
#             shifts[j, i] = torch.tensor([-shift_y, -shift_x])  # Reverse shift for the opposite direction

#     # return shifts
#     # Compute final shifts
#     final_shifts = compute_final_shifts(shifts)
#     final_shifts = torch.cumsum(final_shifts, dim=0)
#     return final_shifts
    
# import networkx as nx
# def compute_final_shifts(pairwise_shifts):
#     # Create a graph where each node is an image and each edge is a shift
#     G = nx.Graph()

#     num_images = pairwise_shifts.shape[0]
#     for i in range(num_images):
#         for j in range(i+1, num_images):
#             shift = pairwise_shifts[i, j]
#             # Add an edge between image i and image j with weight equal to the magnitude of the shift
#             G.add_edge(i, j, weight=torch.norm(shift), shift=shift)

#     # Compute the minimum spanning tree of the graph
#     mst = nx.minimum_spanning_tree(G)

#     # Initialize final shifts with zeros
#     final_shifts = torch.zeros((num_images, 2), device=pairwise_shifts.device)

#     # Use a DFS to compute the shifts of all images relative to the reference
#     for edge in nx.dfs_edges(mst, source=0):
#         i, j = edge
#         shift = mst.edges[i, j]['shift']
#         final_shifts[j] = final_shifts[i] + shift

#     return final_shifts
    
# ### 

# def apply_shifts(moving_images, shifts):
#     # Assuming moving_images is a 3D tensor [num_images, height, width]
#     # and shifts is a 2D tensor [num_images, 2] (y, x)
#     N, H, W = moving_images.shape
#     device = moving_images.device
    

#     # Normalize the shifts to be in the range [-1, 1]
#     shifts = shifts / torch.tensor([H, W]).to(device)

#     # Create a grid of indices
#     grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float), 
#                                     torch.arange(W, device=device, dtype=torch.float)) 

#     # Normalize the grid to be in the range [-1, 1]
#     grid_y = 2.0 * grid_y / (H - 1) - 1.0
#     grid_x = 2.0 * grid_x / (W - 1) - 1.0

#     # Apply the shifts to the grid of indices
#     grid_y = grid_y[None] + shifts[:, 0][:, None, None]
#     grid_x = grid_x[None] + shifts[:, 1][:, None, None]

#     # Stack the grids to create a [N, H, W, 2] grid
#     grid = torch.stack([grid_x, grid_y], dim=-1)

#     # Use the shifted grid of indices to index into moving_images
#     intersection = F.grid_sample(moving_images.unsqueeze(1), grid, align_corners=False)

#     return intersection.squeeze(1)


#turns out that looping over the shifts is faster than using grid_sample on the entire thing, at least on CPU
# @torch.jit.script
def apply_shifts(moving_images, shifts):
    # If shifts is a 1D tensor, add an extra dimension to make it 2D
    if len(shifts.shape) == 1:
        shifts = shifts.unsqueeze(0)

    # print('shifts',shifts.shape)

    N, H, W = moving_images.shape
    device = moving_images.device
    # Normalize the shifts to be in the range [-1, 1]
    shifts = shifts / torch.tensor([H, W]).to(device)

    # Create a grid of indices
    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float), 
                                    torch.arange(W, device=device, dtype=torch.float),
                                    indexing='ij') 

    # Normalize the grid to be in the range [-1, 1]
    grid_y = 2.0 * grid_y / (H - 1) - 1.0
    grid_x = 2.0 * grid_x / (W - 1) - 1.0

    # Initialize tensor to hold the shifted images
    shifted_images = torch.empty_like(moving_images)

    # Find unique shifts and their indices
    unique_shifts, indices = torch.unique(shifts, dim=0, return_inverse=True)

    # Group the indices by their corresponding shifts
    bincounts = torch.bincount(indices)
    split_sizes = [bincounts[i].item() for i in range(bincounts.size(0))]
    grouped_indices = torch.split_with_sizes(indices, split_sizes)

    for i, group in enumerate(grouped_indices):
        # Get the shift for this group
        shift = unique_shifts[i]

        # Apply the shift to the grid of indices
        grid_y_shifted = grid_y[None] + shift[0]
        grid_x_shifted = grid_x[None] + shift[1]

        # Stack the grids to create a [1, H, W, 2] grid
        grid = torch.stack([grid_x_shifted, grid_y_shifted], dim=-1)

        # Use the shifted grid of indices to index into the slices
        shifted_slices = torch.nn.functional.grid_sample(moving_images[group].unsqueeze(1), 
                                                         grid.repeat(len(group),1,1,1), 
                                                         mode='bilinear', #default
                                                         align_corners=False #default
                                                         )

        # Store the shifted slices
        shifted_images[group] = shifted_slices.squeeze(1)

    return shifted_images
    
# def shifts_to_slice(shifts,shape):
#     """
#     Find the minimal crop box from time lapse registration shifts.
#     """
# #     max_shift = np.max(shifts,axis=0)
# #     min_shift = np.min(shifts,axis=0)
# #     slc = tuple([slice(np.maximum(0,0+int(mn)),np.minimum(s,s-int(mx))) for mx,mn,s in zip(np.flip(max_shift),np.flip(min_shift),shape)])
#     # slc = tuple([slice(np.maximum(0,0+int(mn)),np.minimum(s,s-int(mx))) for mx,mn,s in zip(max_shift,min_shift,shape)])
#     upper_shift = np.min(shifts,axis=0)
#     lower_shift = np.max(shifts,axis=0)
#     slc = tuple([slice(np.maximum(0,0+int(l)),np.minimum(s,s-int(u))) for u,l,s in zip(upper_shift,lower_shift,shape)])
#     return slc

    
# from scipy.ndimage import map_coordinates

# def apply_shifts_numpy(moving_images: np.ndarray, shifts: np.ndarray) -> np.ndarray:
#     N, H, W = moving_images.shape

#     # Normalize the shifts to be in the range [-1, 1]
#     shifts = shifts / np.array([H, W])

#     # Create a grid of indices
#     grid_y, grid_x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

#     # Apply the shift to the grid of indices
#     grid_y_shifted = grid_y[None] + shifts[:, 0, None, None]
#     grid_x_shifted = grid_x[None] + shifts[:, 1, None, None]

#     # Use the shifted grid of indices to index into the slices
#     shifted_images = np.empty_like(moving_images)
#     for i, image in enumerate(moving_images):
#         shifted_images[i] = map_coordinates(image, [grid_y_shifted[i], grid_x_shifted[i]], order=1)

#     return shifted_images  