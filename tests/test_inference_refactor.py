import numpy as np
from omnirefactor import models


def test_inference():
    model = models.OmniModel(gpu=False, model_type='bact_phase_omni', net_avg=False,
                             diam_mean=0.0, nclasses=4, dim=2, nchan=1)

    test_image = np.ones((512, 512, 2), dtype=np.float32)
    test_image[3:100, 2:40, 0] = 217
    test_image[225:300, 60:80, 1] = 398

    params = {
        'rescale': None,
        'mask_threshold': -2,
        'flow_threshold': 0,
        'transparency': True,
        'omni': True,
        'cluster': True,
        'resample': True,
        'verbose': False,
        'tile': False,
        'niter': None,
        'augment': False,
        'affinity_seg': False,
    }

    masks, flows, styles = model.eval(test_image, **params)

    assert masks is not None
    assert flows is not None
    assert styles is not None
    assert isinstance(masks, np.ndarray)
    assert isinstance(flows, list)
    assert isinstance(styles, np.ndarray)
    assert masks.shape == (512, 512)
    assert flows[0].shape == (512, 512, 4)
    assert masks.max() > 0
