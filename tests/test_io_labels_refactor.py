import numpy as np

from omnirefactor.io.imio import imwrite
from omnirefactor.io.labels import load_train_test_data
from omnirefactor.io.links import write_links


def test_load_train_test_data_duplicates_single_image(tmp_path):
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    img = np.zeros((16, 16), dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.uint16)
    mask[4:12, 4:12] = 1
    flows = np.zeros((3, 16, 16), dtype=np.float32)

    imwrite(str(train_dir / "img0.tif"), img)
    imwrite(str(train_dir / "img0_masks.tif"), mask)
    imwrite(
        str(train_dir / "img0_flows.tif"),
        flows,
        photometric="minisblack",
        planarconfig="separate",
    )
    write_links(str(train_dir), "img0", {(1, 2)})

    images, labels, links, names, *_ = load_train_test_data(
        str(train_dir),
        image_filter="",
        mask_filter="_masks",
        look_one_level_down=False,
        unet=False,
        omni=False,
        do_links=True,
    )

    assert len(images) == 2
    assert len(labels) == 2
    assert len(links) == 2
    assert len(names) == 2
    assert labels[0].shape[0] == 4  # mask + 3 flows
    assert links[0] == {(1, 2)}


def test_load_train_test_data_omni_skips_flows(tmp_path):
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    img = np.zeros((8, 8), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint16)
    mask[2:6, 2:6] = 1
    flows = np.zeros((3, 8, 8), dtype=np.float32)

    imwrite(str(train_dir / "img1.tif"), img)
    imwrite(str(train_dir / "img1_masks.tif"), mask)
    imwrite(
        str(train_dir / "img1_flows.tif"),
        flows,
        photometric="minisblack",
        planarconfig="separate",
    )

    images, labels, links, names, *_ = load_train_test_data(
        str(train_dir),
        image_filter="",
        mask_filter="_masks",
        look_one_level_down=False,
        unet=False,
        omni=True,
        do_links=False,
    )

    assert len(images) == 2
    assert labels[0].ndim == 2  # flows ignored for omni=True
    assert links[0] is None
