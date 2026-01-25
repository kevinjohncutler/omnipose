import numpy as np

from omnirefactor.io.imio import imwrite
from omnirefactor.io.labels import (
    get_label_files,
    load_train_test_data,
    masks_flows_to_seg,
    save_masks,
)
from omnirefactor.io.links import write_links


def test_load_train_test_data_duplicates_single_image(tmp_path):
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    img = np.zeros((16, 16), dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.uint16)
    mask[4:12, 4:12] = 1
    imwrite(str(train_dir / "img0.tif"), img)
    imwrite(str(train_dir / "img0_masks.tif"), mask)
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
    assert labels[0].ndim == 2
    assert links[0] == {(1, 2)}


def test_load_train_test_data_omni_skips_flows(tmp_path):
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    img = np.zeros((8, 8), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint16)
    mask[2:6, 2:6] = 1
    imwrite(str(train_dir / "img1.tif"), img)
    imwrite(str(train_dir / "img1_masks.tif"), mask)

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
    assert labels[0].ndim == 2
    assert links[0] is None


def test_get_label_files_with_flows_and_links(tmp_path):
    img1 = tmp_path / "imgA.tif"
    img2 = tmp_path / "imgB.tif"
    imwrite(str(img1), np.zeros((4, 4), dtype=np.uint8))
    imwrite(str(img2), np.zeros((4, 4), dtype=np.uint8))

    imwrite(str(tmp_path / "imgA_masks.tif"), np.zeros((4, 4), dtype=np.uint16))
    imwrite(str(tmp_path / "imgA_masks.png"), np.zeros((4, 4), dtype=np.uint16))
    imwrite(str(tmp_path / "imgB_masks.png"), np.zeros((4, 4), dtype=np.uint16))

    write_links(str(tmp_path), "imgA", {(1, 2)})
    write_links(str(tmp_path), "imgB", {(2, 3)})

    labels, links = get_label_files(
        [str(img1), str(img2)],
        label_filter="_masks",
        links=True,
    )
    assert labels[0].endswith("_masks.tif")
    assert labels[1].endswith("_masks.png")
    assert all(p.endswith("_links.txt") for p in links)


def test_get_label_files_ext_override(tmp_path):
    img = tmp_path / "imgD.tif"
    imwrite(str(img), np.zeros((4, 4), dtype=np.uint8))
    labels = get_label_files([str(img)], label_filter="_masks", ext=".npy")
    assert labels[0].endswith("_masks.npy")


def test_masks_flows_to_seg_writes_npy(tmp_path):
    img = np.zeros((8, 8), dtype=np.uint8)
    masks = np.zeros((8, 8), dtype=np.uint16)
    masks[2:6, 2:6] = 1
    flows = [
        np.zeros((8, 8), dtype=np.float32),
        np.zeros((2, 8, 8), dtype=np.float32),
        np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8),
        np.zeros((2, 8, 8), dtype=np.float32),
    ]
    out = tmp_path / "img0.tif"
    masks_flows_to_seg(img, masks, flows, 12.0, str(out))

    seg_path = tmp_path / "img0_seg.npy"
    assert seg_path.exists()
    data = np.load(seg_path, allow_pickle=True).item()
    assert "masks" in data
    assert "outlines" in data
    assert data["masks"].shape == masks.shape


def test_masks_flows_to_seg_list(tmp_path):
    img = np.zeros((8, 8), dtype=np.uint8)
    masks = np.zeros((8, 8), dtype=np.uint16)
    masks[1:5, 1:5] = 1
    flows = [
        np.zeros((8, 8), dtype=np.float32),
        np.zeros((2, 8, 8), dtype=np.float32),
        np.ones((8, 8), dtype=np.float32),
        np.zeros((2, 8, 8), dtype=np.float32),
    ]
    out1 = tmp_path / "img1.tif"
    out2 = tmp_path / "img2.tif"
    masks_flows_to_seg(
        [img, img],
        [masks, masks],
        [flows, flows],
        11.0,
        [str(out1), str(out2)],
    )
    assert (tmp_path / "img1_seg.npy").exists()
    assert (tmp_path / "img2_seg.npy").exists()


def test_save_masks_tif_and_ncolor(tmp_path):
    img = np.zeros((8, 8), dtype=np.uint8)
    masks = np.zeros((8, 8), dtype=np.uint16)
    masks[2:6, 2:6] = 1
    flows = [
        np.zeros((8, 8), dtype=np.uint8),
        np.zeros((2, 8, 8), dtype=np.float32),
    ]
    out = tmp_path / "imgX.tif"
    save_masks(
        img,
        masks,
        flows,
        str(out),
        png=False,
        tif=True,
        save_flows=True,
        save_ncolor=True,
        save_plot=False,
        savedir=str(tmp_path),
    )
    assert (tmp_path / "imgX_cp_masks.tif").exists()
    assert (tmp_path / "imgX_cp_ncolor_masks.png").exists()
    assert (tmp_path / "imgX_flows.tif").exists()
    assert (tmp_path / "imgX_dP.tif").exists()
