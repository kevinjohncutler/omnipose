import torch
from .utils import torch_norm, kernel_setup
from torchvf.losses import ivp_loss
import numpy as np
from .core import steps_batch, _get_affinity_torch, divergence_torch, _gradient
from torchvf.numerics import interp_vf, ivp_solver



class EulerLoss(ivp_loss.IVPLoss):
    def __init__(self,device,dim):
        super().__init__(dx=np.sqrt(dim)/5,                                                   
                         # n_steps=2,
                        #  n_steps=(dim**2)//2,
                         n_steps=dim, # maybe should be dim                                                  
                         device=device,                                                   
                         mode='nearest_batched',
                         # mode='bilinear_batched'
                        )
        
        
# one way to implement affinity loss would be to get euler loss going


class AffinityLoss(torch.nn.Module):

    def __init__(self,device,dim):
        self.device = device
        self.dim = dim
        self.steps, self.inds, self.idx, self.fact, self.sign = kernel_setup(self.dim)
        super().__init__()
        
        self.MSE = torch.nn.MSELoss(reduction='mean')
        self.BCE = torch.nn.BCELoss(reduction='mean')

    def forward(self,flow_pred,dist_pred,flow_gt,dist_gt): # y is GT, x is predicted 
    
        # torch.autograd.set_detect_anomaly(True) # this is a problem on MPS

        mask_threshold = 0
        # foreground must be union of x and y foregrounds             
        foreground = torch.logical_or(dist_pred >= mask_threshold, dist_gt >= mask_threshold)
    
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

        cell_px = (Ellipsis,)+coords[-self.dim:]
        # if niter is None:
        #     niter = int(2*(self.dim+1)*torch.mean(dist_pred[(Ellipsis,)+coords]) / 2)
        niter = 10
        
        # this shoudl be paralleized 
        ags = []
        fps = []
        bds = []
        for f,d in zip([flow_pred,flow_gt],[dist_pred,dist_gt]):
            # final_points = initial_points.clone()
            # final_p, traced_p = steps_batch(initial_points[cell_px],
            #                                         flow_pred/5., #<<<<<<<<<<< add support for other options here 
            #                                         niter=niter, omni=True)
            # final_points[cell_px] = final_p.squeeze()
            vf = interp_vf(f, mode = "nearest_batched")
            final_points = ivp_solver(vf,
                                      initial_points, 
                                        dx = np.sqrt(self.dim)/5,
                                        n_steps = 2,
                                        solver = "euler")[-1] 
            
            fps.append(final_points)

            affinity_graph = _get_affinity_torch(initial_points, 
                                                final_points, 
                                                f/5., #<<<<<<<<<<< add support for other options here 
                                                d, 
                                                foreground, 
                                                self.steps,
                                                self.fact,
                                                niter,
                                                )
            ags.append(affinity_graph*1.0)
            
            csum = torch.sum(affinity_graph,axis=1)
            bds.append(1.0*torch.logical_and(csum<(3**self.dim-1),csum>=self.dim))
        
        # lossA = self.BCE(*ags)
        lossA = self.MSE(*ags)
        lossE = self.MSE(*fps)
        lossB = self.BCE(*bds) # zeor?
        # print(lossA,lossE,lossB)
        return lossA, lossE, lossB
            


class WeightedMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,y,Y,w):
        return torch.mean(torch.square(y-Y)*w)


class SineSquaredLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x,y,w):
        eps = 1e-12
        magX = torch_norm(x,dim=1)
        magY = torch_norm(y,dim=1)
        denom = torch.multiply(magX,magY)
        dot = torch.sum(torch.stack([x[:,k]*y[:,k] for k in range(x.shape[1])],dim=1),dim=1)
        
        # need to handle zero denominator, so if either or both are zero, cos is 1 so loss is minimum (0)
        # so that handles transitions from boundary to background just fine 
        cos = torch.where(denom>eps,dot/(denom+eps),1) 
        return torch.mean((1-cos**2)*w)        
        # return torch.where(mask,(1-cos**2)*w,torch.nan).nanmean() # possible alternative 

        
class NormLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,y,Y,w):
        return torch.mean(torch.square(torch_norm(y,dim=1)-torch_norm(Y,dim=1))*w)/25
        # return torch.mean(torch.square(torch_norm(y,dim=1)-torch_norm(Y,dim=1))*w)/25
        
        # return torch.nn.functional.binary_cross_entropy_with_logits(torch_norm(y,dim=1)/5,torch_norm(Y,dim=1)/5)
        # return torch.nn.functional.l1_loss(torch_norm(y,dim=1)/5,torch_norm(Y,dim=1)/5)

# more efficient combination of the two loss functions, fewer calls to torch_norm 
class SSL_Norm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self,x,y,dist,w,bd): # y is GT, x is predicted 
        eps = 1e-12
        magX = torch_norm(x,dim=1)
        magY = torch_norm(y,dim=1)
        denom = torch.multiply(magX,magY)
        # dot = torch.sum(torch.stack([x[:,k]*y[:,k] for k in range(x.shape[1])],dim=1),dim=1)
        dot = (x * y).sum(dim=1)
        # cossq = torch.where(denom>eps,dot/(denom+eps),1)**2 #golden 
        # cossq = torch.where(denom>0,dot/(denom+eps),1)**2 
        
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
        
        s = torch.square(magY-magX)
        NL = (torch.mean(s*bd)/2+torch.mean(s)/25)/2
        sel = torch.where(bd)
        SSL = torch.mean((1-cossq[sel]))
        
        
        
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
class SSL_Norm_MSE(torch.nn.Module):
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
        
        # or maybe MSE / cos?
        cos_weighted = torch.mean(err / (cos**2+1)) # terrible, causes border merging 
        
        return mse_weighted, cos_weighted, torch.mean(torch.square(magX-magY)*w)/25
        # return torch.mean(w*((1-cos**2)+(1/25)*(magX-magY)**2))
        
        

class DerivativeLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,y,Y,w,mask):
        # y is awlays nbatch x dim x shape, shape is Ly x Lx, Lt x Ly x Lx, or Lz x Ly x Lx. 
        # so y[0] grabs one example
        # axes = [k for k in range(len(y[0]))]     
        # print('shape',y.shape,y[0].shape)
        dim = y.shape[1]
        dims = [k for k in range(-dim,0)]
        # print('dims',dim,dims)
        dy = torch.stack(torch.gradient(y,dim=dims))
        dY = torch.stack(torch.gradient(Y,dim=dims))
        sel = torch.where(mask) # read that masked selection could be causing this 
        return torch.mean(torch.sum(torch.square((dy-dY)/5.),axis=0)[sel]*w[sel])    
    

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


class CorrelationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x,y):
        vx = x - torch.mean(x, dim=0)
        vy = y - torch.mean(y, dim=0)
        num = torch.sum(vx * vy, dim=0)
        denom = torch.sum(vx**2, dim=0) * torch.sum(vy**2, dim=0)
        cost = torch.where(denom>0,num/torch.sqrt(denom),-1)
        return -torch.mean(cost) 