import torch
import numpy as np
import time
from .core import random_crop_warp, masks_to_flows_batch, batch_labels
from .utils import normalize99
from .plot import rgb_flow
import tifffile 

from .utils import get_flip, _taper_mask_ND, unaugment_tiles_ND, average_tiles_ND, make_tiles_ND

# import multiprocessing as mp
# import imageio
from aicsimageio import AICSImage


import torch.nn.functional as F
def torch_zoom(img, scale_factor=1.0, dim=2, size=None, mode='bilinear'):
    # Calculate the target size
    target_size = [int(dim * scale_factor) for dim in img.shape[-dim:]] if size is None else size

    # Use interpolate to resize the image
    img = F.interpolate(img, size=target_size, mode=mode, align_corners=False)

    return img

class eval_loader(torch.utils.data.DataLoader):
    def __init__(self, dataset, model, postprocess_fn, **kwargs):
        super().__init__(dataset, **kwargs)
        self.model = model
        self.postprocess_fn = postprocess_fn

    def __iter__(self):
        for batch in super().__iter__():
            
            print(batch)
            
            # Run the model on the batch
            predictions = self.model._run_net(batch)
            # Perform post-processing on the predictions
            post_processed_predictions = self.postprocess_fn(predictions)
            yield post_processed_predictions

class sampler(torch.utils.data.Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices) 
            
# dataset for evaluation
# incidentally, this also provides a platform for doing image augmentations (combine output from rotated images etc.)

# need to modify this to use multichannel for bact_phase omni...

class eval_set(torch.utils.data.Dataset):
    def __init__(self, data, dim, 
                 channel_axis=None,
                 device=torch.device('cpu'),
                 normalize_stack=True, 
                 rescale=1.0,
                 pad_mode='reflect',
                 interp_mode='bilinear',
                 extra_pad=1,
                 projection=None,
                 tile=False,
                 aics_args=None,
                 contrast_limits=None):
        self.data = data
        self.dim = dim
        self.channel_axis = channel_axis
        self.stack = isinstance(self.data, np.ndarray)
        self.aics =  isinstance(self.data, AICSImage)
        self.aics_args = aics_args if aics_args is not None else {}
        self.list = isinstance(self.data,list)
        if self.list:
            self.files = isinstance(self.data[0], str)
        else:
            self.files = False
        
        self.device = device
        self.normalize_stack = normalize_stack
        self.rescale = rescale
        self.pad_mode = pad_mode
        self.interp_mode = interp_mode
        self.extra_pad = extra_pad
        self.tile = tile
        self.contrast_limits = contrast_limits
        
    def __iter__(self):
        worker_info = mp.get_worker_info()

        if worker_info is None:  # For single-process training
            start = 0
            end = len(self)
        else:  # For multi-process training
            total_samples = len(self)
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            per_worker = int(total_samples / num_workers)
            leftover = total_samples % num_workers
            start = worker_id * per_worker
            end = start + per_worker

            if worker_id == num_workers - 1:
                end += leftover

        for index in range(start, end):
            yield self[index] 

 
#         if imgs.ndim == 1+self.dim:
#             imgs = np.expand_dims(imgs,axis=1)
#         elif not channel_axis:
#             # if the channel axis is 0, then the default order is correct: B,C,Y,X
#             # otherwise, we need to move it to position 1
#             imgs = np.moveaxis(imgs,channel_axis,1)


    def __getitem__(self, inds, no_pad=False, no_rescale=False):
        if isinstance(inds, int):
            inds = [inds]
        
        
        if self.stack:
            # data is in memory, index it
            imgs = torch.tensor(self.data[inds].astype(np.float32), device=self.device)
            
        elif self.list:   
            
            # should call some recursive function here... 
        
            # data is in storage, read it in as a list
            # or if it is a list, concatenate it
            # imgs = [torch.tensor((imread(self.data[index]).astype(float) if self.files else self.data[index].astype(float)),device=self.device) 
            #                     for index in inds]
            
            imgs = [[] for _ in inds]
            
            for i,index in enumerate(inds):
                if self.files:
                    file = self.data[index]
                    # img = AICSImage(file).data.squeeze().astype(float)
                    img = AICSImage(file).get_image_data("YX", out_of_memory=True).squeeze()
                    # print('here',img.shape, AICSImage(file).shape,  AICSImage(file).dims, AICSImage(file).get_image_data("CYX", out_of_memory=True).shape)
                    
                    # img = tifffile.imread()
                else:
                    img = self.data[index]
                    
                imgs[i] = torch.tensor(img.astype(np.float32),device=self.device)
                
            
            # I would like to be able to handle different shapes... perhaps by padding
            # but I do not think that is reasonable for now. 
                        
            # shapes = [img.shape for img in imgs]
            # same_shape = len(set(shapes)) == 1

            # if not same_shape:
            #     # Find the maximum dimensions
            #     max_dims = [max(dim) for dim in zip(*shapes)]

            #     # Pad all images to the maximum dimensions
            #     imgs = [torch.nn.functional.pad(img, (0, 0, max_dims[0] - img.shape[0], max_dims[1] - img.shape[1])) for img in imgs]
            imgs = torch.stack(imgs,dim=0)
            
        elif self.aics:
            kwargs = self.aics_args.copy()
            slice_dim = kwargs.pop('slice_dim')
            kwargs[slice_dim] = inds 
            imgs = self.data.get_image_data(**kwargs).squeeze().astype(float)
            imgs = torch.tensor(imgs,device=self.device)
            
        # print(imgs.shape,torch.tensor(self.data[0].astype(float)).shape,self.data[0].shape,self.stack,self.dim)
        # # add a line here to catch if already a tensor

        # imgs = torch.stack([normalize99(i) for i in imgs]) looks like my normalize99 function is fine...
        # print('fdgfdg',self.channel_axis)
        # print(imgs.shape,imgs.ndim,'aa',self.dim)
        # if imgs.ndim == 1+self.dim:
        
        
        # at this point, we have stacked the images. We need to decide if they are
        # already stacked as a batch or if they need a batch dimension.
        # We also need to add the channel dimension.
        
        # could assume no channels
        # or I could assume that the list/stack has more than one entry
        # so it will always have spatial dims, a stack dim, and maybe a channel dim
        
        # wait, maybe my 3D data uis not training right, as it seems to not run with both channel and batch dims
        
        # if dimension matches ndim, then no channel axis exists 
        if imgs.ndim == self.dim:
            imgs = imgs.unsqueeze(0) # add channel dim
            print('adding channel dim')
        
       # if the channel axis exists 
        if self.channel_axis is not None:
        #     if self.channel_axis!=0:
        # elif not self.channel_axis:
        
            dims = [0, self.channel_axis] + list(range(1, self.channel_axis)) + list(range(self.channel_axis+1, imgs.ndim))
            print('d',dims,len(dims),imgs.shape)
            imgs = imgs.permute(dims)
        else:
            imgs = imgs.unsqueeze(1) 
        

        if not self.tile:
            # images are normalized per tile, so no need to normalize at once here, 
            # which throws an error if the tensor it too big 
            imgs = normalize99(imgs,
                               contrast_limits=self.contrast_limits,
                               dim=None if self.normalize_stack else 0) # much faster on GPU now
        
        
        # ADD RESCALE CODE HERE?
        if self.rescale is not None and self.rescale != 1.0 and not no_rescale:
            imgs = torch_zoom(imgs, self.rescale, mode=self.interp_mode)
            
            
        if no_pad:
            return imgs.squeeze()
        else:
            shape = imgs.shape[-self.dim:]
            div = 16 
            extra = self.extra_pad
            idxs = [k for k in range(-self.dim,0)]
            Lpad = [int(div * np.ceil(shape[i]/div) - shape[i]) for i in idxs]
            lower_pad = [extra*div//2 + Lpad[k]//2 for k in range(self.dim)] # lower pad along each axis
            upper_pad = [extra*div//2 + Lpad[k] - Lpad[k]//2 for k in range(self.dim)] # upper pad along each axis 

            # for torch.nn.functional.pad(), we need (x1,x2,y1,y2,...) where x1,x2 are the bounds for the LAST dimension
            # and y1,y2 are bounds for the penultimate etc., and we already computed upper and lower bound lists pad1, pad2 
            # thus, we need to reverse the order of the dimensions and assemble this tuple 
            pads = tuple()
            for k in range(self.dim):
                pads += (lower_pad[-(k+1)],upper_pad[-(k+1)])

            subs = [np.arange(lower_pad[k],lower_pad[k]+shape[k]) for k in range(self.dim)]
                        
            # mode = 'constant'
            # print(imgs.shape,'ggg')
            # # value = torch.mean(imgs) if mode=='constant' else 0  
            # value = 0 # turns out performance on sparse cells much better if padding is zero 
            I = torch.nn.functional.pad(imgs, pads, mode=self.pad_mode, value=None)            
            return I, inds, subs
            
    def _run_tiled(self, batch, model, 
                   batch_size=8, augment=False, bsize=224, 
                   normalize=True, 
                   tile_overlap=0.1, return_conv=False):
    
        for imgi in batch:
            IMG, subs, shape, inds = make_tiles_ND(imgi,
                                                bsize=bsize,
                                                augment=augment,
                                                normalize=normalize,
                                                tile_overlap=tile_overlap) 
            # IMG now always returned in the form (ny*nx, nchan, ly, lx) 
            # for either tiling or augmenting
            
            niter = int(np.ceil(IMG.shape[0] / batch_size))
            nout = model.nclasses + 32*return_conv
            y = torch.zeros((IMG.shape[0], nout)+tuple(IMG.shape[-model.dim:]),device=IMG.device)
            for k in range(niter):
                irange = np.arange(batch_size*k, min(IMG.shape[0], batch_size*k+batch_size))
                y0 = model.net(IMG[irange])[0]
                arg = (len(irange),)+y0.shape[-(model.dim+1):]
                y[irange] = y0.reshape(arg)

            if augment: 
                y = unaugment_tiles_ND(y, inds, model.unet)
            yf = average_tiles_ND(y, subs, shape) #<<<
            slc = tuple([slice(s) for s in shape])
            yf = yf[(Ellipsis,)+slc]
            return yf

    
    def collate_fn(self,worker_data):
        # worker_data is a list of batches from each worker
        # Each batch is a tuple of (imgi, labels, scale)

        # Separate the batches from each worker
        worker_imgs, worker_inds, worker_subs = zip(*worker_data)

        # Concatenate the worker batches along the batch dimension
        batch_imgs = torch.cat(worker_imgs, dim=0)
        batch_inds = [item for sublist in worker_inds for item in sublist]
        batch_subs = [item for sublist in worker_subs for item in sublist]

        return batch_imgs.float(), batch_inds, batch_subs

    def __len__(self):
        return len(self.data)
    
    import torch
    
    
# does not need getitem, 
# class eval_set(torch.utils.data.IterableDataset):
#     def __init__(self, data):
#         self.data = data
#         self.stack = isinstance(self.data, np.ndarray)
#         self.files = isinstance(self.data[0], str)

#     def __iter__(self):
#         worker_info = torch.utils.data.get_worker_info()

#         if worker_info is None:  # For single-process training
#             start = 0
#             end = len(self)
#         else:  # For multi-process training
#             total_samples = len(self)
#             num_workers = worker_info.num_workers
#             worker_id = worker_info.id
#             per_worker = int(total_samples / num_workers)
#             leftover = total_samples % num_workers
#             start = worker_id * per_worker
#             end = start + per_worker

#             if worker_id == num_workers - 1:
#                 end += leftover

#         for index in range(start, end):
#             if self.stack:
#                 imgs = self.data[index]
#             else:
#                 imgs = imageio.imread(self.data[index]) if self.files else self.data[index]

#             print('yyy',imgs.shape)
#             yield torch.tensor(imgs.astype(float))
            
            
            
            
#         # pad it
#         div = 16  # Set your desired div value here
#         extra = 1  # Set your desired extra value here
#         dim = 2  # Set your desired dim value here

#         # Determine the size of the padded tensor
#         inds = [k for k in range(-dim,0)]
#         Lpad = [int(div * np.ceil(self.data[0].shape[i]/div) - self.data[0].shape[i]) for i in inds]
#         pad1 = [extra*div//2 + Lpad[k]//2 for k in range(dim)]
#         pad2 = [extra*div//2 + Lpad[k] - Lpad[k]//2 for k in range(dim)]

#         shape = self.data[0].shape[:-dim] + tuple([self.data[0].shape[i] + pad1[k] + pad2[k] for i, k in zip(inds, range(dim))])


#     def __len__(self):
#         return len(self.data)


### training dataset

class train_set(torch.utils.data.Dataset):
# class dataset(torch.utils.data.IterableDataset):

# class dataset(torch.utils.data.TensorDataset):

    def __init__(self, data, labels, links, timing=False, **kwargs):
        
        # make these params into attributes
        self.__dict__.update(kwargs)
        
        self.data = data
        self.labels = labels
        self.links = links
        self.timing = timing
        
        if not hasattr(self,'augment'):
            self.augment = True
            
        # random_rotate_and_resize setup now goes here 
        if self.tyx is None:
            n = 16
            kernel_size=2
            base = kernel_size
            L = max(round(224/(base**4)),1)*(base**4) # rounds 224 up to the right multiple to work for base 
            # not sure if 4 downsampling or 3, but the "multiple of 16" elsewhere makes me think it must be 4, 
            # but it appears that multiple of 8 actually works?
            self.tyx = (L,)*self.dim if self.dim==2 else (8*n,)+(8*n,)*(self.dim-1) #must be divisible by 2**3 = 8
        
        self.scale_range = max(0, min(2, float(self.scale_range)))
        self.do_flip = True
        self.dist_bg = 5
        self.smooth = False # smoothing while iterating, much slower to converge 
        self.normalize = False # barely needs normalizing now... having low mag in the center helps 
        self.gamma_range = [.75,2.5]
        self.nimg = len(data)
        self.rescale = self.diam_train / self.diam_mean if self.rescale else np.ones(self.nimg, np.float32)
        
        self.v1 = [0]*(self.dim-1)+[1]
        self.v2 = [0]*(self.dim-2)+[1,0]
                

    def __iter__(self):
        worker_info = mp.get_worker_info()

        if worker_info is None:  # For single-process training
            start = 0
            end = len(self)
        else:  # For multi-process training
            total_samples = len(self)
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            per_worker = int(total_samples / num_workers)
            leftover = total_samples % num_workers
            start = worker_id * per_worker
            end = start + per_worker

            if worker_id == num_workers - 1:
                end += leftover

        for index in range(start, end):
            yield self[index]

        
    # def collate_fn(self,data):
    #     imgs,labels,inds = zip(*data)
    #     return torch.cat(imgs,dim=0),torch.cat(labels,dim=0),inds
    
    def collate_fn(self,worker_data):
        # worker_data is a list of batches from each worker
        # Each batch is a tuple of (imgi, labels, scale)

        # Separate the batches from each worker
        worker_imgs, worker_labels, worker_inds = zip(*worker_data)

        # Concatenate the worker batches along the batch dimension
        batch_imgs = torch.cat(worker_imgs, dim=0)
        batch_labels = torch.cat(worker_labels, dim=0)
        # batch_inds = torch.cat(worker_inds, dim=0)
        batch_inds = [item for sublist in worker_inds for item in sublist]

        return batch_imgs, batch_labels, batch_inds

    
    def worker_init_fn(self,worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        
        
    def __len__(self):
        return self.nimg

    
    def __getitem__(self, inds):
        # this is called with a batchsampler, so always has a list of inds, not one index
        if self.timing:
            tic = time.time()
        
#         print(inds)
#         if type(inds)!=list:
#             inds = [inds]
        nimg = len(inds)
        imgi = np.zeros((nimg, self.nchan)+self.tyx, np.float32)
        labels = np.zeros((nimg,)+self.tyx, np.float32)
        scale = np.zeros((nimg,self.dim), np.float32)
        links = [self.links[idx] for idx in inds]
        

        for i,idx in enumerate(inds):
            imgi[i], labels[i], scale[i] = random_crop_warp(img=self.data[idx], 
                                                            Y=self.labels[idx],
                                                            tyx = self.tyx,
                                                            v1=self.v1,
                                                            v2=self.v2,
                                                            nchan=self.nchan, 
                                                            rescale=self.rescale[idx], 
                                                            scale_range=self.scale_range, 
                                                            gamma_range=self.gamma_range, 
                                                            do_flip=self.do_flip, 
                                                            ind=idx,
                                                            augment=self.augment,
                                                           )
            # print('hey', self.data[idx].shape,self.labels[idx].shape)
            
            
        out = masks_to_flows_batch(labels, links,
                                   device=self.device,
                                   omni=self.omni,
                                   dim=self.dim,
                                   affinity_field=self.affinity_field
                                  )

        X = out[:-4]
        slices = out[-4]
        affinity_graph = out[-1]
        masks,bd,T,mu = [torch.stack([x[(Ellipsis,)+slc] for slc in slices]) for x in X]
        lbl = batch_labels(masks,
                           bd,
                           T,
                           mu,
                           self.tyx,
                           dim=self.dim,
                           nclasses=self.nclasses,
                           device=self.device
                          )
        
        # mucat = torch.concatenate(tuple(lbl[:,-self.dim:]),dim=-1)
        # rgb = rgb_flow(mucat.swapaxes(0,1)[:,-2:]).cpu().numpy()
        # tifffile.imwrite('/home/kcutler/DataDrive/teresa_high_frame_rate/crops/edited/testlabelaug{}.tif'.format(inds),np.concatenate(lbl[:,0].cpu().numpy(),axis=-1))
        # tifffile.imwrite('/home/kcutler/DataDrive/teresa_high_frame_rate/crops/edited/testdistaug{}.tif'.format(inds),np.concatenate(lbl[:,3].cpu().numpy(),axis=-1))
        # tifffile.imwrite('/home/kcutler/DataDrive/teresa_high_frame_rate/crops/edited/testphaseaug{}.tif'.format(inds),np.concatenate(imgi.squeeze(),axis=-1))
        # tifffile.imwrite('/home/kcutler/DataDrive/teresa_high_frame_rate/crops/edited/testflowaug{}.tif'.format(inds),rgb)
             
        
        imgi = torch.tensor(imgi,device=self.device)
        # imgi = torch.tensor(imgi).to(self.device,non_blocking=True) # slower
        if self.timing:
            print('single image augmentation time: {:.2f}, Device: {}'.format(time.time()-tic,self.device))
        
   
        return imgi, lbl, inds

