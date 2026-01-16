import os
import numpy as np
import torch
from omnirefactor import models


def test_training(tmp_path):
    model = models.OmniModel(gpu=False, omni=True, nclasses=4, nchan=2, diam_mean=0)

    train_images = [np.ones((512, 512), dtype=np.float32) for _ in range(2)]
    train_masks = [np.zeros((512, 512), dtype=np.int32) for _ in range(2)]
    train_links = [[(1, 2), (3, 4)], [(5, 6), (7, 8)]]

    for i in range(2):
        train_images[i][3:100, 2:40] = 217
        train_images[i][225:300, 60:80] = 398
        train_masks[i][3:100, 2:40] = 1

    save_dir = tmp_path

    model.train(
        train_images,
        train_masks,
        train_links,
        save_every=2,
        n_epochs=2,
        batch_size=1,
        save_path=str(save_dir),
        channels=[0, 0],
        min_train_masks=1,
        rescale=0,
    )

    model_path = None
    models_dir = os.path.join(save_dir, "models")
    for file_name in os.listdir(models_dir):
        if file_name.startswith("._"):
            continue
        if file_name.endswith("epoch_1"):
            model_path = os.path.join(models_dir, file_name)
            break

    assert model_path is not None, "Model checkpoint for epoch 1 not saved"

    model_file = torch.load(model_path, weights_only=False, map_location="cpu")

    assert isinstance(model_file, dict)
    assert any(
        key.startswith("downsample") or key.startswith("output") for key in model_file.keys()
    )
