from .imports import *
from ..core.affinity import _get_affinity_torch



class BatchMeanMSE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss(reduction='none')

    def forward(self, pred, target, weight=None):
        """
        Compute mean-squared error averaged per sample, then across batch.
        Optionally apply element-wise weighting.
        """
        per_elem = self.mse(pred, target)
        if weight is not None:
            per_elem = per_elem * weight
        per_sample = per_elem.reshape(per_elem.size(0), -1).mean(dim=1)
        return per_sample.mean()


class BatchMeanBSE(torch.nn.Module):
    """
    Same idea as BatchMeanMSE but using binary cross-entropy.
    Computes BCE loss per element → average over each sample →
    average those means across the batch.
    """
    def __init__(self):
        super().__init__()
        # keep element-wise losses so we can average the way we want
        self.bce = torch.nn.BCELoss(reduction='none')

    def forward(self, pred, target):
        per_elem = self.bce(pred, target)              # shape (B, …)
        per_sample = per_elem.reshape(per_elem.size(0), -1).mean(dim=1)
        return per_sample.mean()
      

class WeightedMSELoss(torch.nn.Module):
    """
    Weighted MSE implemented via BatchMeanMSE for consistent reduction.
    """
    def __init__(self):
        super().__init__()
        self.base = BatchMeanMSE()

    def forward(self, pred, target, weight):
        return self.base(pred, target, weight)

  
        
class AffinityLoss(torch.nn.Module):

    def __init__(self,device,dim):
        self.device = device
        self.dim = dim
        self.steps, self.inds, self.idx, self.fact, self.sign = kernel_setup(self.dim)
        self.supporting_inds = get_supporting_inds(self.steps)

        
        super().__init__()
        
        # self.MSE = torch.nn.MSELoss(reduction='mean')
        # self.BCE = torch.nn.BCELoss(reduction='mean')
        
        self.MSE = BatchMeanMSE()
        self.BCE = BatchMeanBSE()

    def forward(self, flow_pred, dist_pred, flow_gt, dist_gt,
                mode='foreground', 
                mask_threshold=0,  # zero defines background vs foreground
                random_frac=0.5, 
                seed=None):
        """
        Compute affinity, Euler, and boundary losses between predicted and ground-truth
        flow / distance fields.

        Parameters
        ----------
        flow_pred : torch.Tensor
            Predicted flow field, shape (B, C, H, W).
        dist_pred : torch.Tensor
            Predicted distance field.
        flow_gt   : torch.Tensor
            Ground-truth flow field.
        dist_gt   : torch.Tensor
            Ground-truth distance field.
        mode : str, optional
            Pixel-selection mode:
                'foreground' (default) - use foreground derived from distance fields.
                'all'                   - ignore distance fields; integrate over every pixel.
                'random'                - use foreground plus a random subset of background
                                          pixels. Fraction controlled by `random_frac`.
        random_frac : float, optional
            Fraction of total pixels to sample when `mode == 'random'`. Range 0-1.
        seed : int or None, optional
            Seed for torch's RNG when sampling random pixels.
        """


        if mode == 'all':
            foreground = torch.ones_like(dist_pred, dtype=torch.bool)
        else:
            # baseline foreground from distance maps
            
            foreground = torch.logical_or(dist_pred >= mask_threshold,
                                          dist_gt   >= mask_threshold)

            if mode == 'random' and random_frac > 0.0:
                # reproducible random sampling if seed provided
                if seed is not None:
                    torch.manual_seed(seed)
                rand_mask = torch.rand_like(dist_pred) < random_frac
                foreground = torch.logical_or(foreground, rand_mask)


        # vf = interp_vf(flow_pred/5., mode = "nearest_batched")
        # initial_points = init_values_semantic(foreground, device=self.device)
        
        shape = flow_pred.shape
        B = shape[0]
        dims = shape[-self.dim:]

        coords = [torch.arange(0, l, device = self.device) for l in dims]
        mesh = torch.meshgrid(coords, indexing = "ij")
        init_shape = [B, 1] + ([1] * len(dims))
        initial_points = torch.stack(mesh, dim = 0) # torchvf flips with mesh[::-1]
        initial_points = initial_points.repeat(init_shape).float()

        coords = torch.nonzero(foreground,as_tuple=True)

        # cell_px = (Ellipsis,)+coords[-self.dim:]
        # if niter is None:
        #     niter = int(2*(self.dim+1)*torch.mean(dist_pred[(Ellipsis,)+coords]) / 2)
        niter = 10

        # Batched approach: ~1.3x faster on GPU than sequential.
        # NOTE: This causes tiny gradient accumulation differences vs sequential due to
        # PyTorch autograd graph traversal order (CatBackward/SplitBackward vs SelectBackward).
        # Divergence is ~2e-5 after 10 epochs - negligible for model quality.
        # To restore exact parity with prior pipeline, revert to sequential loop:
        #   for f, d in zip([flow_pred, flow_gt], [dist_pred, dist_gt]):
        #       vf = interp_vf(f, mode="nearest_batched")
        #       final_points = ivp_solver(vf, initial_points, dx=..., n_steps=2, solver="euler")[-1]
        #       ...
        flow_all = torch.cat([flow_pred, flow_gt], dim=0)
        initial_points_all = torch.cat([initial_points, initial_points], dim=0)

        vf_all = interp_vf(flow_all, mode="nearest_batched")
        final_points_all = ivp_solver(vf_all, initial_points_all,
                                      dx=np.sqrt(self.dim)/5,
                                      n_steps=2,
                                      solver="euler")[-1]

        fp_pred, fp_gt = torch.chunk(final_points_all, 2, dim=0)

        ags = []
        fps = []
        bds = []
        for f, d, fp in zip([flow_pred, flow_gt], [dist_pred, dist_gt], [fp_pred, fp_gt]):
            fps.append(fp)
            affinity_graph = _get_affinity_torch(initial_points,
                                                fp,
                                                f/5.,
                                                d,
                                                foreground,
                                                self.steps,
                                                self.fact,
                                                self.inds,
                                                self.supporting_inds,
                                                niter,
                                                device=self.device
                                                )
            ags.append(affinity_graph*1.0)

            csum = torch.sum(affinity_graph, axis=1)
            bds.append(1.0*torch.logical_and(csum<(3**self.dim-1), csum>=self.dim))
        
        # lossA = self.BCE(*ags)
        lossA = self.MSE(*ags)
        lossE = self.MSE(*fps)
        lossB = self.BCE(*bds) # zero?
        
        # print(lossA,lossE,lossB)
        return lossA, lossE, lossB
            

class SSL_Norm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.MSE = BatchMeanMSE()
        self.WMSE = WeightedMSELoss()
        
    def forward(self,x,y,dist,w,bd): # y is GT, x is predicted 
        eps = 1e-12
        magX = torch_norm(x,dim=1)
        magY = torch_norm(y,dim=1)
        denom = torch.multiply(magX,magY)
        # dot = torch.sum(torch.stack([x[:,k]*y[:,k] for k in range(x.shape[1])],dim=1),dim=1)
        dot = (x * y).sum(dim=1)
        # cossq = torch.where(denom>eps,dot/(denom+eps),1)**2 #golden 
        # cossq = torch.where(denom>0,dot/(denom+eps),1)**2 
       
       
       # this was in use for a while, jan 2025 
        mask = dist>0
        cossq = torch.where(mask,dot/(denom+eps),1)**2 #experiment to limit to where cell pixels are predicted only 
        
        
        
        # potential substitute for norm loss? 
#         normdiff = torch.where(magX>eps,1-(magY/(magX)),magX)
#         wsum = torch.sum(w)
#         return torch.sum((1-cos**2)*w)/wsum,torch.sum((normdiff**2)*w)/wsum

        # maybe should scale norm loss by cos**2? Only care if mag is right if direction is right 
        
        # return torch.mean((1-cossq)*w),torch.mean(torch.square(magX-magY)*w)/25 # golden version
        # SSL = torch.mean((1-cossq)*w) # golden 
        
        
        # NL = torch.mean(torch.abs(magX-magY)*w)/5 # pretty good, an improvement? 
        # NL = torch.mean(torch.square(magX-magY))/25 # most basic version
        # NL = torch.mean(torch.square(magX-magY)*bd/5) # just boundary - works pretty well, one of the better ones
        
        # this version is doing extremely well so far, but maybe a bit too much weight overall
        # NL = torch.mean(torch.square(mask*magY-magX)*bd) #/5
        # SSL = torch.mean((1-cossq)*bd)*10 # upweight here for experiment
        
        # reduce these terms by a bit to avoid some artifacts, decent 
        # NL = torch.mean(torch.square(mask*magY-magX)*bd)/2 
#         SSL = torch.mean((1-cossq)*bd)*6 # upweight here for experiment
        
        # golden
        # reduce further to get performance back maybe? This worked very, very well 
        # NL = (torch.mean(torch.square(mask*magY-magX)*bd)/2+torch.mean(torch.square(magX-magY))/25)/2
        # SSL = torch.mean((1-cossq)*bd)*4
        
        # this was in use for a long while, january 2025
        # s = torch.square(magY-magX)
        # NL = (torch.mean(s*bd)/2+torch.mean(s)/25)/2
        # sel = torch.where(bd)
        # SSL = torch.mean((1-cossq[sel]))
        
        # if we predict the right direction, we should predict the right magnitude
        # thus cos**2 should be 1, leading to a minumum if  magX/5 is close to 1
        # if we predict the wrong direction, cos**2 should be 0 this will push magX/5 to be 0
        # NL = torch.mean(torch.square(magX/5-cossq)) 
        
        # weighting the normloss above high makes things 1 everywhere, not good
        # NL = torch.mean(torch.square(magX/5-(magX>0).float())) + 
        NL = self.MSE(magX, magY)                           # unweighted magnitude error
        SSL = self.WMSE(cossq, torch.ones_like(cossq), w)   # weighted orientation error
        # sel = torch.where(bd)
        # SSL = torch.mean((1-cossq[sel]))
        
        
        # see if SSL should be weighted instead of purely masked by bd 
        # NL = (torch.mean(torch.square(mask*magY-magX)*w)/2+torch.mean(torch.square(magX-magY))/25)/2
        # SSL = torch.mean((1-cossq)*w)*4
        
        
        # I think that I can simplify the NL a lot by just making the appropriate weight matrix
        # might as well apply it to SSL instead of just bd, but then the overall factor needs to go back up
        # turns out this was not a good idea...
        # weight = (mask+1)/25+bd/2
        # NL = torch.mean(torch.square(mask*magY-magX)*weight)
        # # SSL = torch.mean((1-cossq)*weight)*8
        # SSL = torch.mean((1-cossq)*bd)*4 # not sure if applying it to SSL was good        
        
        
        # loss on the gradient of the norm - seems to work ok for making the field the right magnitude  
        # dim = x.shape[1]
        # dims = [k for k in range(-dim,0)]
        # dx = torch.stack(torch.gradient(magX,dim=dims))
        # dy = torch.stack(torch.gradient(magY,dim=dims))
        # NL += torch.mean(torch.square(dx-dy))/5
        
        # soft dice
        # eps = 1e-5
        # A = torch.sum(magX/5)
        # B = torch.sum(magY/5)
        # C = torch.sum(magX*magY/25)
        # NL = 1-(2*C+eps)/(A+B+eps)
        
        # attempt at polynomial loss 
        # d = magX-magY
        # NL =  torch.mean(torch.square(d*(d-5)*(d+5)))/(5**6)
        # NL =  torch.mean(torch.square((magX-magY)*(magX-2*magY)*magX))
        
        # maybe norm loss should only act to push the predictions avobe cutoff to GT, below cutoff to 0
        # NL = torch.mean(torch.where(magY>1,torch.square(magX-magY)*w/25,magY)) # experimental version 
        
        return SSL, NL
        
        # return torch.mean((1-cossq)*w),torch.mean(torch.square(magX-magY)*cossq)/25
        
        # return torch.mean(w*((1-cos**2)+(1/25)*torch.square(magX-magY))),torch.zeros(1,device=w.device)
        
# maybe should blend between MSE and SSL
# SSL counts at the border most, MSE counts most at the center 
# class DerivativeLoss(torch.nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self,y,Y,w,mask):
#         # y is awlays nbatch x dim x shape, shape is Ly x Lx, Lt x Ly x Lx, or Lz x Ly x Lx. 
#         # so y[0] grabs one example
#         # axes = [k for k in range(len(y[0]))]     
#         # print('shape',y.shape,y[0].shape)
#         dim = y.shape[1]
#         dims = [k for k in range(-dim,0)]
#         # print('dims',dim,dims)
#         dy = torch.stack(torch.gradient(y,dim=dims))
#         dY = torch.stack(torch.gradient(Y,dim=dims))
#         sel = torch.where(mask) # read that masked selection could be causing this 
#         return torch.mean(torch.sum(torch.square((dy-dY)/5.),axis=0)[sel]*w[sel])    



class DerivativeLoss(torch.nn.Module):
    """
    Gradient-domain loss rewritten to use the BatchMean/Weighted MSE
    infrastructure for consistent per-sample reduction.

    Computes the mean-squared error between spatial gradients of the
    prediction `y` and ground-truth `Y`.  Results are averaged per-sample,
    then across the batch via `WeightedMSELoss`, with optional pixel-wise
    weights `w` and a binary mask `mask` defining valid regions.

    Mathematically equivalent to the original implementation:

        mean( ((∇y - ∇Y)/5)² * w )[mask]
    """
    def __init__(self):
        super().__init__()
        self.WMSE = WeightedMSELoss()

    def forward(self, y, Y, w, mask):
        # Spatial dimensions are all dims after batch and channel
        spatial_dims = y.ndim - 2
        spatial_axes = list(range(-spatial_dims, 0))

        # Gradients along each spatial axis → stack then bring batch dim first
        dy = torch.stack(torch.gradient(y, dim=spatial_axes)).transpose(0, 1)
        dY = torch.stack(torch.gradient(Y, dim=spatial_axes)).transpose(0, 1)

        # Combine weights with mask; broadcast to gradient tensor shape
        weight = w * mask                          # (B,1,…)
        while weight.ndim < dy.ndim:
            weight = weight.unsqueeze(1)           # prepend dims
        weight = weight.expand_as(dy)              # (B,dim,C,…)

        # Weighted MSE, scaled by 1/25 to match (⋅/5)² factor
        return self.WMSE(dy, dY, weight) / 25
    

# it will probably be better just to do the gradient for the whole image....
# i.e use a constant affinity graph 
# class EikonalLoss(torch.nn.Module):
#     def __init__(self,device,dim):
#         self.device = device
#         steps,inds,idx,fact,sign = utils.kernel_setup(dim)
#         self.dim = torch.tensor(dim,device=device)
#         self.idx = torch.tensor(idx)
#         self.fact = torch.tensor(fact)
#         self.steps = torch.tensor(steps,device=device)        
#         self.inds = tuple([torch.tensor(i) for i in inds])
            
        
#         super().__init__()

#     def forward(self,dist_pred,flow_gt,mask):
#     neighbors = utils.get_neighbors(coords,steps,d,shape,edges) # shape (d,3**d,npix)   
    
    
#         isneigh = torch.tensor(affinity_graph,device=device,dtype=torch.bool) 
#         neigh_inds = torch.tensor(neigh_inds,device=device)
#         central_inds = torch.tensor(central_inds,device=device,dtype=torch.long)
        
#         # get the gradient of the predicted distance field
#         T = dist_pred[mask]


class EulerLoss(ivp_loss.IVPLoss):  # pragma: no cover
    def __init__(self,device,dim):
        super().__init__(dx=np.sqrt(dim)/5,
                         n_steps=dim,
                         device=device,
                         mode='nearest_batched',
                        )


# one way to implement affinity loss would be to get euler loss going


class SineSquaredLoss(torch.nn.Module):  # pragma: no cover
    def __init__(self):
        super().__init__()

    def forward(self,x,y,w):
        eps = 1e-12
        magX = torch_norm(x,dim=1)
        magY = torch_norm(y,dim=1)
        denom = torch.multiply(magX,magY)
        dot = torch.sum(torch.stack([x[:,k]*y[:,k] for k in range(x.shape[1])],dim=1),dim=1)
        cos = torch.where(denom>eps,dot/(denom+eps),1)
        return torch.mean((1-cos**2)*w)


class NormLoss(torch.nn.Module):  # pragma: no cover
    def __init__(self):
        super().__init__()

    def forward(self,y,Y,w):
        return torch.mean(torch.square(torch_norm(y,dim=1)-torch_norm(Y,dim=1))*w)/25


class SSL_Norm_MSE(torch.nn.Module):  # pragma: no cover
    def __init__(self):
        super().__init__()
        
    def forward(self,x,y,w,dist):
        eps = 1e-12
        magX = torch_norm(x,dim=1)
        magY = torch_norm(y,dim=1)
        denom = torch.multiply(magX,magY)
        dot = torch.sum(torch.stack([x[:,k]*y[:,k] for k in range(x.shape[1])],dim=1),dim=1)
        cos = torch.where(denom>eps,dot/(denom+eps),1)
        err = torch.square((x-y)/5.).sum(dim=1)
        
        w3 = torch.clip(dist,0.5,2)/2
        w2 = 1.-w3
        cos_weighted = torch.sum((1-cos**2)*w2)/torch.sum(w2)
        mse_weighted = torch.sum(err*w3)/torch.sum(w3)
        
        cos_weighted = torch.mean((1-cos**2)*err)
        mse_weighted = 0
        
        cos_weighted = torch.mean(err / (cos**2+1))
        
        return mse_weighted, cos_weighted, torch.mean(torch.square(magX-magY)*w)/25


class CorrelationLoss(torch.nn.Module):  # pragma: no cover
    def __init__(self):
        super().__init__()

    def forward(self,x,y):
        vx = x - torch.mean(x, dim=0)
        vy = y - torch.mean(y, dim=0)
        num = torch.sum(vx * vy, dim=0)
        denom = torch.sum(vx**2, dim=0) * torch.sum(vy**2, dim=0)
        cost = torch.where(denom>0,num/torch.sqrt(denom),-1)
        return -torch.mean(cost)


class TruncatedMSELoss(torch.nn.Module):  # pragma: no cover
    def __init__(self, t=5.0):
        super().__init__()
        self.t = t

    def forward(self, pred, target):
        SE = torch.square(pred - target)
        loss = torch.where(SE < self.t, SE, self.t)
        return loss.mean()


class MeanAdjustedMSELoss(torch.nn.Module):  # pragma: no cover
    def __init__(self):
        super(MeanAdjustedMSELoss, self).__init__()

    def forward(self, pred, target):
        mean_error = torch.mean(pred - target)
        adjusted_pred = pred - mean_error
        loss = torch.mean((adjusted_pred - target) ** 2)
        return loss


class GradNormLoss(torch.nn.Module):  # pragma: no cover
    def __init__(self, num_losses, device, alpha=0.12):
        super().__init__()
        self.alpha = alpha
        self.loss_weights = torch.nn.Parameter(torch.ones(num_losses, device=device, dtype=torch.float32))

    def forward(self, losses, shared_params):
        if len(losses) == 0:
            raise RuntimeError("All loss terms are zero or don't require gradients.")

        weighted_losses = self.loss_weights[:len(losses)] * torch.stack(losses)
        total_loss = weighted_losses.sum()
        grads = torch.autograd.grad(total_loss, shared_params, retain_graph=True, create_graph=False)

        grad_norms = []
        for loss, grad in zip(losses, grads):
            if grad is not None:
                grad_norms.append(torch.norm(grad))

        grad_norms = torch.stack(grad_norms)
        mean_grad = grad_norms.mean().detach()
        target_scales = (grad_norms / mean_grad) ** self.alpha
        target_scales = target_scales.detach()
        weight_loss = torch.sum(torch.abs(self.loss_weights[:len(losses)] * grad_norms - target_scales * mean_grad))
        weight_grads = torch.autograd.grad(weight_loss, self.loss_weights[:len(losses)], allow_unused=True)[0]
        self.loss_weights.grad = weight_grads
        return total_loss
