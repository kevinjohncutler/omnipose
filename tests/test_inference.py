import numpy as np
from cellpose_omni import models

def test_inference():

    model = models.CellposeModel(gpu=False, model_type='bact_omni')
                             diam_mean=diam_mean, nclasses=nclasses, dim=dim, nchan=nchan)

    
    test_image = np.ones((512, 512, 2), dtype=np.float32)
    test_image[3:100, 2:40, 0] = 217
    test_image[225:300, 60:80, 1] = 398
    
    chans = [0,0] 
    mask_threshold = -1 
    verbose = 0
    transparency = True # transparency in flow output
    rescale= None # give this a number if you need to upscale or downscale your images
    flow_threshold = 0 # default is .4, but only needed if there are spurious masks to clean up; slows down output
    resample = True #whether or not to run dynamics on rescaled grid or original grid 
    cluster = True
    omni = True
    masks, flows, styles = model.eval(test_image,channels=chans,rescale=rescale,mask_threshold=mask_threshold,
                                               transparency=transparency,flow_threshold=flow_threshold,omni=omni, 
                                               resample=resample,verbose=verbose, cluster=cluster,interp=True)
    
    assert masks is not None, "Masks output is None"
    assert flows is not None, "Flows output is None"
    assert styles is not None, "Styles output is None"
    assert isinstance(masks, np.ndarray), f"Expected masks to be a numpy array, but got {type(masks)}"
    assert isinstance(flows, list), f"Expected flows to be a list, but got {type(flows)}"
    assert isinstance(styles, np.ndarray), f"Expected styles to be a numpy array, but got {type(styles)}"
    assert masks.shape == (512, 512)
    assert flows.shape == (512, 512, 2)
    assert styles.shape == (512, 512, 2)
    assert masks.max() > 0