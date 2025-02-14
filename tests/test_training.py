import numpy as np
from cellpose_omni import models

def test_training():
    model = models.CellposeModel(gpu=False, model_type='bact_phase_omni', net_avg=False, 
                                 diam_mean=0., nclasses=4, dim=2, nchan=1)

    train_images = [np.ones((512, 512), dtype=np.float32) for _ in range(2)]
    train_masks = [np.zeros((512, 512), dtype=np.int32) for _ in range(2)]
    
    for i in range(2):
        train_images[i][3:100, 2:40, 0] = 217
        train_images[i][225:300, 60:80, 0] = 398
        train_masks[i][3:100, 2:40, 0] = 1
    
    params = {
        'learning_rate': 0.001,
        'n_epochs': 2,
        'batch_size': 1,
        'rescale': None,
        'save_path': './',
        'save_every': 1
    }
    
    model.train(train_data=train_images, train_labels=train_masks, train_links=[], **params)
    
    import os
    assert os.path.exists('./cellpose_model.pth'), "Model file was not saved"