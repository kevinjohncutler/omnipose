import types

import numpy as np
import pytest

from omnipose.cli import runner


def test_confirm_prompt_yes(monkeypatch):
    answers = iter(["y"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert runner.confirm_prompt("Proceed?") is True


def test_confirm_prompt_no(monkeypatch):
    answers = iter(["n"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert runner.confirm_prompt("Proceed?") is False


def test_resolve_channels():
    args = types.SimpleNamespace(nchan=2, all_channels=False, chan=1, chan2=2)
    assert runner._resolve_channels(args) == [1, 2]
    args = types.SimpleNamespace(nchan=1, all_channels=True, chan=1, chan2=2)
    assert runner._resolve_channels(args) is None


def test_resolve_model_builtin(monkeypatch):
    calls = {}

    class DummyModel:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setattr(runner.models, "MODEL_NAMES", ["cyto"])
    monkeypatch.setattr(runner.models, "OmniModel", DummyModel)

    args = types.SimpleNamespace(
        pretrained_model="cyto",
        omni=False,
        use_gpu=False,
        fast_mode=False,
        no_net_avg=False,
        nclasses=2,
        logits=False,
        nsample=4,
        nchan=1,
        dim=2,
    )
    runner._resolve_model(args, device="cpu", norm_type="batch")
    assert calls["model_type"] == "cyto"


def test_resolve_model_missing_path(monkeypatch):
    calls = {}

    class DummyModel:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setattr(runner.models, "MODEL_NAMES", [])
    monkeypatch.setattr(runner.models, "OmniModel", DummyModel)
    monkeypatch.setattr(runner.os.path, "exists", lambda _: False)

    args = types.SimpleNamespace(
        pretrained_model="missing",
        omni=False,
        use_gpu=False,
        fast_mode=True,
        no_net_avg=False,
        nclasses=2,
        logits=False,
        nsample=4,
        nchan=1,
        dim=2,
    )
    runner._resolve_model(args, device="cpu", norm_type="batch")
    assert calls["model_type"] == "cyto"


def test_run_evaluation_no_images(monkeypatch):
    monkeypatch.setattr(runner.io, "get_image_files", lambda *a, **k: [])
    args = types.SimpleNamespace(
        tyx=None,
        testing=True,
        save_png=False,
        save_tif=False,
        save_flows=False,
        save_ncolor=False,
        use_gpu=False,
        gpu_number=None,
        pretrained_model="cyto",
        omni=False,
        fast_mode=False,
        no_net_avg=False,
        nclasses=2,
        logits=False,
        nsample=4,
        nchan=1,
        dim=2,
        all_channels=False,
        chan=0,
        chan2=0,
        dir=".",
        mask_filter="_masks",
        img_filter="",
        look_one_level_down=False,
        diameter=0,
        rescale=None,
        do_3D=False,
        no_resample=False,
        flow_threshold=0.4,
        mask_threshold=0,
        niter=None,
        diam_threshold=12,
        invert=False,
        batch_size=1,
        no_interp=False,
        cluster=False,
        no_suppress=False,
        channel_axis=None,
        z_axis=None,
        affinity_seg=False,
        anisotropy=1.0,
        verbose=False,
        min_size=15,
        max_size=None,
        transparency=False,
        exclude_on_edges=False,
        no_npy=True,
        dir_above=False,
        savedir=None,
        in_folders=False,
    )
    with pytest.raises(ValueError):
        runner._run_evaluation(args)


def test_run_evaluation_saves(monkeypatch):
    image = np.zeros((8, 8), dtype=np.float32)

    class DummyModel:
        diam_mean = 30

        def eval(self, *a, **k):
            masks = np.zeros((8, 8), dtype=np.int32)
            flows = [np.zeros((2, 8, 8), dtype=np.float32)]
            return masks, flows

    monkeypatch.setattr(runner.models, "assign_device", lambda *a, **k: ("cpu", False))
    monkeypatch.setattr(runner, "_resolve_model", lambda *a, **k: DummyModel())
    monkeypatch.setattr(runner.io, "get_image_files", lambda *a, **k: ["img.tif"])
    monkeypatch.setattr(runner.io, "imread", lambda *a, **k: image)
    monkeypatch.setattr(runner.io, "masks_flows_to_seg", lambda *a, **k: None)
    monkeypatch.setattr(runner.io, "save_masks", lambda *a, **k: None)
    monkeypatch.setattr(runner.utils, "clean_boundary", lambda *a, **k: np.zeros((8, 8), dtype=np.int32))
    monkeypatch.setattr(runner.utils.fastremap, "renumber", lambda *a, **k: None)

    args = types.SimpleNamespace(
        tyx=None,
        testing=True,
        save_png=True,
        save_tif=False,
        save_flows=False,
        save_ncolor=False,
        use_gpu=False,
        gpu_number=None,
        pretrained_model="cyto",
        omni=False,
        fast_mode=False,
        no_net_avg=False,
        nclasses=2,
        logits=False,
        nsample=4,
        nchan=1,
        dim=2,
        all_channels=False,
        chan=0,
        chan2=0,
        dir=".",
        mask_filter="_masks",
        img_filter="",
        look_one_level_down=False,
        diameter=0,
        rescale=None,
        do_3D=False,
        no_resample=False,
        flow_threshold=0.4,
        mask_threshold=0,
        niter=None,
        diam_threshold=12,
        invert=False,
        batch_size=1,
        no_interp=False,
        cluster=False,
        no_suppress=False,
        channel_axis=None,
        z_axis=None,
        affinity_seg=False,
        anisotropy=1.0,
        verbose=False,
        min_size=15,
        max_size=None,
        transparency=False,
        exclude_on_edges=True,
        no_npy=False,
        dir_above=False,
        savedir=None,
        in_folders=False,
    )
    runner._run_evaluation(args)


@pytest.mark.parametrize("diameter", [15, 30])
def test_run_evaluation_with_diameter(monkeypatch, diameter):
    image = np.zeros((8, 8), dtype=np.float32)
    called = {}

    class DummyModel:
        diam_mean = 30

        def eval(self, *a, **k):
            called["diameter"] = k.get("diameter")
            masks = np.zeros((8, 8), dtype=np.int32)
            flows = [np.zeros((2, 8, 8), dtype=np.float32)]
            return masks, flows

    monkeypatch.setattr(runner.models, "assign_device", lambda *a, **k: ("cpu", False))
    monkeypatch.setattr(runner, "_resolve_model", lambda *a, **k: DummyModel())
    monkeypatch.setattr(runner.io, "get_image_files", lambda *a, **k: ["img.tif"])
    monkeypatch.setattr(runner.io, "imread", lambda *a, **k: image)
    monkeypatch.setattr(runner.io, "masks_flows_to_seg", lambda *a, **k: None)
    monkeypatch.setattr(runner.io, "save_masks", lambda *a, **k: None)

    args = types.SimpleNamespace(
        tyx=None,
        testing=True,
        save_png=False,
        save_tif=False,
        save_flows=False,
        save_ncolor=False,
        use_gpu=False,
        gpu_number=None,
        pretrained_model="cyto",
        omni=False,
        fast_mode=False,
        no_net_avg=False,
        nclasses=2,
        logits=False,
        nsample=4,
        nchan=1,
        dim=2,
        all_channels=False,
        chan=0,
        chan2=0,
        dir=".",
        mask_filter="_masks",
        img_filter="",
        look_one_level_down=False,
        diameter=diameter,
        rescale=None,
        do_3D=False,
        no_resample=False,
        flow_threshold=0.4,
        mask_threshold=0,
        niter=None,
        diam_threshold=12,
        invert=False,
        batch_size=1,
        no_interp=False,
        cluster=False,
        no_suppress=False,
        channel_axis=None,
        z_axis=None,
        affinity_seg=False,
        anisotropy=1.0,
        verbose=False,
        min_size=15,
        max_size=None,
        transparency=False,
        exclude_on_edges=False,
        no_npy=True,
        dir_above=False,
        savedir=None,
        in_folders=False,
    )
    runner._run_evaluation(args)
    assert called["diameter"] == diameter


def test_run_training_batches(monkeypatch):
    monkeypatch.setattr(runner.models, "assign_device", lambda *a, **k: ("cpu", False))
    monkeypatch.setattr(runner.io, "load_train_test_data",
                        lambda *a, **k: (
                            [np.zeros((8, 8))], [np.zeros((8, 8))], None, ["img"],
                            [], [], None, []
                        ))

    class DummyModel:
        def train(self, *a, **k):
            return "model_path"

    monkeypatch.setattr(runner, "_resolve_model", lambda *a, **k: DummyModel())

    args = types.SimpleNamespace(
        tyx=None,
        batch_size=1,
        train_size=False,
        unet=False,
        use_gpu=False,
        gpu_number=None,
        dir=".",
        test_dir=[],
        img_filter="",
        mask_filter="_masks",
        look_one_level_down=False,
        omni=False,
        links=False,
        no_save=True,
        learning_rate=0.2,
        n_epochs=1,
        channel_axis=None,
        save_every=100,
        save_each=False,
        diameter=0,
        num_workers=0,
        min_train_masks=1,
        logits=False,
        RAdam=False,
        nclasses=2,
        nsample=4,
        nchan=1,
        dim=2,
        fast_mode=False,
        no_net_avg=False,
        pretrained_model="cyto",
        timing=False,
        amp=False,
        affinity_field=False,
        tensorboard=False,
        sym_kernels=False,
        symmetry_weight=1.0,
    )
    runner._run_training(args)
    assert args.batch_size == 2


def test_main_calls_training(monkeypatch):
    called = {}

    def _train(args):
        called["train"] = True

    monkeypatch.setattr(runner, "_run_training", _train)
    argv = ["--train", "--dir", ".", "--n_epochs", "1", "--batch_size", "2", "--seed", "1", "--deterministic"]
    runner.main(argv)
    assert called.get("train") is True
