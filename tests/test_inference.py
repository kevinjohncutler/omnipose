import numpy as np
from omnipose import models


def test_inference():
    model = models.OmniModel(gpu=False, model_type='bact_phase_omni', net_avg=False,
                             diam_mean=0.0, nclasses=4, dim=2, nchan=1)

    # Single-channel 2D image with structure the model can process
    test_image = np.zeros((128, 128), dtype=np.float32)
    test_image[3:40, 2:32] = 1.0
    test_image[60:90, 40:70] = 0.8

    params = {
        'rescale_factor': None,
        'mask_threshold': -2,
        'flow_threshold': 0,
        'transparency': False,
        'omni': True,
        'cluster': True,
        'resample': True,
        'verbose': False,
        'tile': False,
        'niter': None,
        'augment': False,
        'affinity_seg': False,
    }

    masks, flows = model.eval(test_image, **params)

    assert masks is not None
    assert flows is not None
    assert isinstance(masks, np.ndarray)
    assert isinstance(flows, list)
    assert masks.ndim >= 2  # at least (H, W) or (1, H, W)
    # Spatial dims should match input
    assert masks.shape[-2:] == (128, 128) or masks.shape[-1] == 128
