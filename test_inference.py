import numpy as np
from cellpose_omni import models

def test_inference():
    # Initialiser le modèle Cellpose
    model = models.CellposeModel(gpu=False, model_type='bact_phase_omni', net_avg=False, 
                                 diam_mean=0., nclasses=4, dim=2, nchan=1)

    # Créer une image de test avec 2 canaux
    test_image = np.ones((512, 512, 2), dtype=np.float32)
    test_image[3:100, 2:40, 0] = 217
    test_image[225:300, 60:80, 1] = 398
    
    params = {
        'rescale': None,  # upscale or downscale your images, None = no rescaling 
        'mask_threshold': -2,  # erode or dilate masks with higher or lower values between -5 and 5 
        'flow_threshold': 0,  # default is .4, but only needed if there are spurious masks to clean up; slows down output
        'transparency': True,  # transparency in flow output
        'omni': True,  # we can turn off Omnipose mask reconstruction, not advised 
        'cluster': True,  # use DBSCAN clustering
        'resample': True,  # whether or not to run dynamics on rescaled grid or original grid 
        'verbose': False,  # turn on if you want to see more output 
        'tile': False,  # average the outputs from flipped (augmented) images; slower, usually not needed 
        'niter': None,  # default None lets Omnipose calculate # of Euler iterations (usually <20) but you can tune it for over/under segmentation 
        'augment': False,  # Can optionally rotate the image and average network outputs, usually not needed 
        'affinity_seg': False  # new feature, stay tuned...
    }

    # Exécuter la fonction d'inférence
    masks, flows, styles = model.eval(test_image, **params)
    
    # Vérifiez que les sorties sont correctes
    assert masks is not None, "Masks output is None"
    assert flows is not None, "Flows output is None"
    assert styles is not None, "Styles output is None"
    assert isinstance(masks, np.ndarray), f"Expected masks to be a numpy array, but got {type(masks)}"
    assert isinstance(flows, list), f"Expected flows to be a list, but got {type(flows)}"
    assert isinstance(styles, np.ndarray), f"Expected styles to be a numpy array, but got {type(styles)}"
    assert masks.shape == (512, 512)
    assert flows[0].shape == (512, 512, 4)
    assert masks.max() > 0