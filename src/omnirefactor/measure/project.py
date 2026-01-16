from .imports import *

def project_to_skeletons(images,labels,augmented_affinity, device, interp, 
                         use_gpu, omni, reference, interp_skel=0, n_step=None,log=True):

    shape = labels.shape
    d = labels.ndim
    neighbors = augmented_affinity[:d]
    affinity_graph = augmented_affinity[d] #.astype(bool) VERY important to cast to bool, now done internally 
    idx = affinity_graph.shape[0]//2
    coords = neighbors[:,idx]
    
    # need ind_matrix to select
    npix = neighbors.shape[-1]
    indexes = np.arange(npix)
    ind_matrix = -np.ones(shape,int)
    ind_matrix[tuple(coords)] = indexes

    # T = masks_to_flows_torch(labels, 
    #                         affinity_graph=affinity_graph, 
    #                         coords=tuple(coords), 
    #                         device=device,
    #                         return_flows=False)[0]

    T, mu = masks_to_flows_torch(labels, 
                            affinity_graph=affinity_graph, 
                            coords=tuple(coords), 
                            device=device)
    
    dt = T.cpu().numpy()
    niter = int(diameters(labels,dt))
    inds = np.array(coords).astype(np.int32)
    p, inds, _ = follow_flows(mu, dt, inds, niter=niter, interp=interp,
                                use_gpu=use_gpu, device=device, omni=omni,
                                suppress=1, calc_trace=0, verbose=0)


    initial = inds
    # final = np.round(p[(Ellipsis,)+tuple(inds)]).astype(int)
    final = p[(Ellipsis,)+tuple(inds)]
        

    # get the skeletons
    # inner = dt>2
    # skel = skeletonize(inner, method='lee')
    skel = skeletonize(labels, dt_thresh=2, dt=dt)
    

    # label skeletons and compute affinity for parametrization
    skel_labels = (skel>0)*labels    
    skel_coords = np.nonzero(skel_labels)
    dim = skel.ndim
    steps, inds, idx, fact, sign = kernel_setup(dim)
    skel_affinity = masks_to_affinity(skel_labels, skel_coords, 
                                      steps, inds, idx, fact, sign, dim)

    # print('00 ',skel_labels.shape,skel_affinity.shape)
    
    # parametrize the skeletons 
    contour_map, contour_list, unique_L = get_contour(skel_labels,
                                                      skel_affinity,
                                                      cardinal_only=0)

    
    # generate mapping from pixel clusters to skeleton 
    # doing per-cell is probably combinatorially faster than all clusters to all skels
    # in which case, clustering step not needed 
    
    images += [dt] # add distance field as a channel
    nchan = len(images)

    N = []
    for c in contour_list:
        if log and n_step is not None:
            n = int(np.ceil(np.log(np.count_nonzero(len(c)))*n_step/np.log(n_step)))+1
        else:
            n = len(c) if n_step is None else n_step
        N.append(n)
    
    projections = [np.zeros((nchan,n)) for c,n in zip(contour_list,N)] # 


    print('c0',len(contour_list),unique_L)
    for contour, L, proj in zip(contour_list, unique_L, projections):
        N = proj.shape[-1]
        # target = np.nonzero(skel_labels==L)
        target = np.array([c[contour] for c in skel_coords])

        # alt: make intermp
        # pts = np.stack([c[contour] for c in skel_coords]).T #<<< using skel coords here
        if interp_skel:
            pts = target.T
            tck, u = splprep(pts.T, u=None, s=len(pts)/6, per=0) # per=1 would be cyclic, not the case here 
            u_new = np.linspace(u.min(), u.max(), N)
            new_pts = splev(u_new, tck, der=0)
            target = np.stack(new_pts)
    
        
        # fix orientation by tracking a pole... 
        # this could break with fast pole movement
        start = target[:,0]
        stop = target[:,-1]
        if reference is not None:
            if np.sum((start-reference)**2,axis=0) > np.sum((stop-reference)**2,axis=0):
                target = target[:,::-1]
                start = stop
        
        # mask_coords = np.nonzero(np.logical_and(labels==L,inner))
        # mask_coords = np.nonzero(labels==L)
        # source_inds = ind_matrix[mask_coords]
        source_inds = np.nonzero(labels[tuple(coords)] == L)[0]
        mask_coords = coords[:,source_inds]

        
        print('AA',np.any(source_inds<0),mask_coords.shape,coords.shape, 
              source_inds.shape,labels.shape)
        
        source = tuple(final[:,source_inds])        
        mapping = project_points(source,target)

        
        # print(source.shape,mapping.shape)
        print('cc',target[0].shape,N)
        
        for c in range(nchan):
            projection = np.zeros(N)
            data = images[c][tuple(mask_coords)]
            print('bb',projection.shape,mapping.shape,data.shape)
            np.add.at(projection,mapping,data)
            counts = np.bincount(mapping, minlength=N)
            projection = safe_divide(projection,counts)
            proj[c] = projection
            # print(np.min(counts),np.max(counts),N)
            # print(np.median(counts))
            # proj[c] = counts[contour]
    
            
    return projections, contour_map, contour_list, skel_coords, start
    

from sklearn.neighbors import NearestNeighbors
def project_points(source, target):
    target = np.array(target)
    source = np.array(source)
    
    source_count = source.shape[1]
    target_count = target.shape[1]
    result = np.empty(source_count, dtype=np.int64)
    
    # Create a k-d tree from the target points
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(target.T)
    
    # Query the k-d tree for the closest target point for each source point
    _, indices = nbrs.kneighbors(source.T)
    
    # Assign the indices of the closest target points to the result array
    result = np.squeeze(indices)
    
    return result
