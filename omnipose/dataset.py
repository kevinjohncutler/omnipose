import torch
import numpy as np
import time
from .core import random_crop_warp, masks_to_flows_batch, batch_labels

import ctypes
import multiprocessing as mp

class dataset(torch.utils.data.Dataset):
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
            # but it appears that multiple of 8 actually works? maybe the n=16 above conflates my experiments in 3D
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

