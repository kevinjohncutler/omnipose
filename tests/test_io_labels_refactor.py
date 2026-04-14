import numpy as np

import omnirefactor.io.labels as labels_mod
from omnirefactor.io import imwrite
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


def test_get_label_files_dir_above_and_parent(tmp_path):
    root = tmp_path / "root"
    images = root / "images"
    labels_dir = root / "labels"
    images.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    img = images / "imgZ.tif"
    imwrite(str(img), np.zeros((4, 4), dtype=np.uint8))
    imwrite(str(labels_dir / "imgZ_masks.tif"), np.zeros((4, 4), dtype=np.uint16))

    labels = get_label_files([str(img)], label_filter="_masks", dir_above=True, subfolder="labels")
    assert labels[0].endswith("_masks.tif")

    labels2 = get_label_files([str(img)], label_filter="_masks", parent=str(labels_dir))
    assert labels2[0].endswith("_masks.tif")


def test_get_label_files_warns_on_missing(tmp_path):
    img = tmp_path / "img_missing.tif"
    imwrite(str(img), np.zeros((4, 4), dtype=np.uint8))
    labels = get_label_files([str(img)], label_filter="_masks")
    assert labels == []


def test_load_train_test_data_with_test_dir(tmp_path):
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    train_dir.mkdir()
    test_dir.mkdir()
    img = np.zeros((8, 8), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint16)
    mask[2:6, 2:6] = 1
    imwrite(str(train_dir / "img0.tif"), img)
    imwrite(str(train_dir / "img0_masks.tif"), mask)
    imwrite(str(test_dir / "imgT.tif"), img)
    imwrite(str(test_dir / "imgT_masks.tif"), mask)

    out = load_train_test_data(
        str(train_dir),
        test_dir=str(test_dir),
        image_filter="",
        mask_filter="_masks",
        look_one_level_down=False,
        unet=False,
        omni=False,
        do_links=False,
    )
    images, labels, links, names, test_images, test_labels, test_links, image_names_test = out
    assert len(images) >= 1
    assert len(test_images) == 1
    assert len(test_labels) == 1
    assert len(image_names_test) == 1


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


def test_masks_flows_to_seg_3d_and_channels(tmp_path):
    img = np.zeros((2, 8, 8), dtype=np.uint8)
    masks = np.zeros((2, 8, 8), dtype=np.uint16)
    masks[:, 2:6, 2:6] = 1
    flows = [
        np.zeros((2, 8, 8), dtype=np.float32),
        np.zeros((2, 8, 8), dtype=np.float32),
        np.zeros((8, 8), dtype=np.float32),
        np.zeros((2, 8, 8), dtype=np.float32),
    ]
    out = tmp_path / "img3d.tif"
    masks_flows_to_seg(img, masks, flows, 12.0, str(out), channels=[0])
    assert (tmp_path / "img3d_seg.npy").exists()


def test_masks_flows_to_seg_flow3d_and_multi_channels(tmp_path):
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    masks = np.zeros((4, 4), dtype=np.uint16)
    masks[1:3, 1:3] = 1
    flows = [
        np.zeros((3, 4, 4), dtype=np.float32),
        np.zeros((2, 4, 4), dtype=np.float32),
        np.ones((4, 4), dtype=np.float32),
        np.zeros((2, 4, 4), dtype=np.float32),
    ]
    out = tmp_path / "imgflow.tif"
    masks_flows_to_seg(
        [img, img, img],
        [masks, masks, masks],
        [flows, flows, flows],
        [10.0, 12.0, 14.0],
        [str(out), str(tmp_path / "imgflow2.tif"), str(tmp_path / "imgflow3.tif")],
        channels=[[0, 1, 2], [0, 1, 2], [0, 1, 2]],
    )
    assert (tmp_path / "imgflow_seg.npy").exists()


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


def test_save_masks_dir_above_and_in_folders(tmp_path):
    parent = tmp_path / "parent"
    img_dir = parent / "images"
    img_dir.mkdir(parents=True)
    img = np.zeros((8, 8), dtype=np.uint8)
    masks = np.zeros((8, 8), dtype=np.uint16)
    masks[2:6, 2:6] = 1
    flows = [
        np.zeros((8, 8), dtype=np.uint8),
        np.zeros((2, 8, 8), dtype=np.float32),
    ]
    out = img_dir / "imgY.tif"
    save_masks(
        img,
        masks,
        flows,
        str(out),
        png=False,
        tif=True,
        in_folders=True,
        dir_above=True,
        suffix="test",
        save_plot=False,
    )
    saved = parent / "masks" / "imgY_cp_masks_test.tif"
    assert saved.exists()



def test_save_masks_png_overflow_warning(tmp_path):
    img = np.zeros((8, 8), dtype=np.uint8)
    masks = np.zeros((8, 8), dtype=np.uint32)
    masks[2:6, 2:6] = 70000
    flows = [
        np.zeros((8, 8), dtype=np.uint8),
        np.zeros((2, 8, 8), dtype=np.float32),
    ]
    out = tmp_path / "imgOverflow.tif"
    save_masks(
        img,
        masks,
        flows,
        str(out),
        png=True,
        tif=False,
        save_plot=False,
    )
    assert (tmp_path / "imgOverflow_cp_masks.tif").exists()


def test_save_masks_png_error_for_3d(tmp_path):
    img = np.zeros((2, 8, 8), dtype=np.uint8)
    masks = np.zeros((2, 8, 8), dtype=np.uint16)
    masks[:, 2:6, 2:6] = 1
    flows = [
        np.zeros((2, 8, 8), dtype=np.uint8),
        np.zeros((2, 2, 8, 8), dtype=np.float32),
    ]
    out = tmp_path / "img3d.tif"
    try:
        save_masks(img, masks, flows, str(out), png=True, tif=False)
    except ValueError:
        assert True
    else:
        assert False, "Expected ValueError for 3D PNG outputs"


def test_save_masks_list_input(tmp_path):
    img = np.zeros((8, 8), dtype=np.uint8)
    masks = np.zeros((8, 8), dtype=np.uint16)
    masks[2:6, 2:6] = 1
    flows = [
        np.zeros((8, 8), dtype=np.uint8),
        np.zeros((2, 8, 8), dtype=np.float32),
    ]
    out1 = tmp_path / "imgL1.tif"
    out2 = tmp_path / "imgL2.tif"
    save_masks(
        [img, img],
        [masks, masks],
        [flows, flows],
        [str(out1), str(out2)],
        png=False,
        tif=True,
        save_plot=False,
    )
    assert (tmp_path / "imgL1_cp_masks.tif").exists()


def test_save_masks_3d_sets_tif(tmp_path):
    img = np.zeros((2, 8, 8), dtype=np.uint8)
    masks = np.zeros((2, 8, 8), dtype=np.uint16)
    masks[:, 2:6, 2:6] = 1
    flows = [
        np.zeros((2, 8, 8), dtype=np.uint8),
        np.zeros((2, 2, 8, 8), dtype=np.float32),
    ]
    out = tmp_path / "img3d_ok.tif"
    save_masks(img, masks, flows, str(out), png=True, tif=True, save_plot=False)
    assert (tmp_path / "img3d_ok_cp_masks.tif").exists()
