import torch
import numpy as np
import time
from .core import random_crop_warp, masks_to_flows_batch, batch_labels
from .utils import normalize99
# import multiprocessing as mp
# import imageio

# This will 
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

class eval_set(torch.utils.data.Dataset):
    def __init__(self, data, dim, channel_axis=0,device=torch.device('cpu'),normalize_stack=True):
        self.data = data
        self.dim = dim
        self.channel_axis = channel_axis
        self.stack = isinstance(self.data, np.ndarray)
        self.files = isinstance(self.data[0],str)
        self.device=device
        self.normalize_stack=normalize_stack
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

#     def __getitem__(self, inds, no_pad=False):
        
#         if isinstance(inds, int):
#             inds = [inds]
            
#         if self.stack:
#             imgs = self.data[inds].astype(float)#.to(self.device)
# #         else:   
# #             # imgs = [(np.asarray(imageio.imread(self.data[index]) if self.files else self.data[index])).astype(float) for index in inds]
# #             imgs = np.stack([(imageio.imread(self.data[index]) if self.files else self.data[index]) for index in inds]).astype(float)
            
#         if imgs.ndim == 1+self.dim:
#             imgs = np.expand_dims(imgs,axis=1)
#         elif not channel_axis:
#             # if the channel axis is 0, then the default order is correct: B,C,Y,X
#             # otherwise, we need to move it to position 1
#             imgs = np.moveaxis(imgs,channel_axis,1)
        
#         imgs = normalize99(imgs) # normalizing as a stack only makes sense for homogeneous datasets, might need a toggle 
        
#         if no_pad:
#             return imgs.squeeze()
#         else:
#             shape = imgs.shape[-self.dim:]
#             div = 16
#             extra = 1
#             idxs = [k for k in range(-self.dim,0)]
#             Lpad = [int(div * np.ceil(shape[i]/div) - shape[i]) for i in idxs]
#             pad1 = [extra*div//2 + Lpad[k]//2 for k in range(self.dim)]
#             pad2 = [extra*div//2 + Lpad[k] - Lpad[k]//2 for k in range(self.dim)]

#             emptypad = tuple([[0,0]]*(imgs.ndim-self.dim))
#             pads = emptypad+tuple(np.stack((pad1,pad2),axis=1))
#             subs = [np.arange(pad1[k],pad1[k]+shape[k]) for k in range(self.dim)]

#             mode = 'reflect'
#             I = np.pad(imgs,pads, mode=mode)

#             return torch.tensor(I), inds, subs

    def __getitem__(self, inds, no_pad=False):
        if isinstance(inds, int):
            inds = [inds]
            
        if self.stack:
            imgs = self.data[inds].astype(float)
        else:   
            imgs = torch.stack([(imageio.imread(self.data[index]) if self.files else self.data[index]) for index in inds]).astype(float)
            
 
        imgs = torch.tensor(imgs, device=self.device)

        # imgs = torch.stack([normalize99(i) for i in imgs]) looks like my normalize99 function is fine...
        
        if imgs.ndim == 1+self.dim:
            imgs = imgs.unsqueeze(1)
            # imgs = torch.cat([imgs,torch.zeros_like(imgs)],dim=1)
            # print('k,jj')
        elif not channel_axis:
            imgs = imgs.permute([0, channel_axis] + list(range(1, channel_axis)) + list(range(channel_axis+1, imgs.ndim)))
        
        imgs = normalize99(imgs,dim=None if self.normalize_stack else 0) # much faster on GPU now

        if no_pad:
            return imgs.squeeze()
        else:
            shape = imgs.shape[-self.dim:]
            div = 16 
            extra = 1
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

            mode = 'reflect'
            # mode = 'constant'
            # # value = torch.mean(imgs) if mode=='constant' else 0  
            # value = 0 # turns out performance on sparse cells much better if padding is zero 
            I = torch.nn.functional.pad(imgs, pads, mode=mode,value=None)
            
            return I, inds, subs

    
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
                                                            ind=idx
                                                           )

        out = masks_to_flows_batch(labels, links,
                                   device=self.device,
                                   omni=self.omni,
                                   dim=self.dim,
                                   affinity_field=self.affinity_field
                                  )[:-2]

        X = out[:-1]
        slices = out[-1]
        masks,bd,T,mu = [torch.stack([x[(Ellipsis,)+slc] for slc in slices]) for x in X]
        lbl = batch_labels(masks,bd,T,mu,
                           self.tyx,
                           dim=self.dim,
                           nclasses=self.nclasses,
                           device=self.device
                          )
        imgi = torch.tensor(imgi,device=self.device)
        # imgi = torch.tensor(imgi).to(self.device,non_blocking=True) # slower
        if self.timing:
            print('single image augmentation time: {:.2f}, Device: {}'.format(time.time()-tic,self.device))
        return imgi, lbl, inds

