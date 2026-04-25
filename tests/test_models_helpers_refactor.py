import os

from omnipose.models import helpers


def test_resolve_pretrained_model_from_name(monkeypatch, tmp_path):
    called = []

    def fake_model_path(model_type, model_index, use_torch):
        called.append((model_type, model_index, use_torch))
        return os.fspath(tmp_path / f"{model_type}_{model_index}")

    monkeypatch.setattr(helpers, "model_path", fake_model_path)

    pretrained, model_name, net_avg, updates, residual_on, style_on, concatenation = (
        helpers.resolve_pretrained_model(
            pretrained_model=["missing/path"],
            model_type="bact_phase_omni",
            net_avg=True,
            use_torch=True,
            model_names=["bact_phase_omni"],
            bd_model_names=["bact_phase_omni"],
            c2_model_names=["bact_phase_omni"],
            omni=True,
        )
    )

    assert model_name == "bact_phase_omni"
    assert net_avg is False
    assert residual_on is True
    assert style_on is True
    assert concatenation is False
    assert pretrained[0].endswith("bact_phase_omni_0")
    assert updates["nclasses"] == 3
    assert updates["nchan"] == 2
    assert "nclasses" in updates


def test_resolve_model_init_config_fallback_cyto():
    config = helpers.resolve_model_init_config(
        pretrained_model=["missing/path"],
        model_type=None,
        net_avg=True,
        model_names=["cyto"],
        bd_model_names=[],
        c2_model_names=[],
        omni=False,
        pretrained_model_exists=False,
    )
    assert config["model_name"] == "cyto"
    assert config["model_indices"] == [0, 1, 2, 3]
    assert config["warn_bad_path"] is True


def test_resolve_model_init_config_no_model_type():
    config = helpers.resolve_model_init_config(
        pretrained_model=None,
        model_type=None,
        net_avg=True,
        model_names=["cyto"],
        bd_model_names=[],
        c2_model_names=[],
        omni=False,
        pretrained_model_exists=True,
    )
    assert config["model_name"] is None
    assert config["model_indices"] == []


def test_resolve_model_init_config_nuclear():
    config = helpers.resolve_model_init_config(
        pretrained_model=["missing/path"],
        model_type="nuclei",
        net_avg=True,
        model_names=["nuclei"],
        bd_model_names=[],
        c2_model_names=[],
        omni=False,
        pretrained_model_exists=False,
    )
    assert config["model_name"] == "nuclei"
    assert config["net_avg"] is True
    assert config["updates"]["diam_mean"] == 17.0
    assert config["updates"]["nclasses"] == 2
    assert config["model_indices"] == [0, 1, 2, 3]


def test_resolve_model_init_config_c2_channel_update():
    config = helpers.resolve_model_init_config(
        pretrained_model=["missing/path"],
        model_type="cyto",
        net_avg=True,
        model_names=["cyto"],
        bd_model_names=[],
        c2_model_names=["cyto"],
        omni=False,
        pretrained_model_exists=False,
    )
    assert config["updates"]["nchan"] == 2


def test_resolve_pretrained_model_keeps_existing_path(monkeypatch, tmp_path):
    model_file = tmp_path / "custom_model"
    model_file.write_text("ok")

    def fail_model_path(*args, **kwargs):
        raise AssertionError("model_path should not be called when path exists")

    monkeypatch.setattr(helpers, "model_path", fail_model_path)

    pretrained, model_name, net_avg, updates, residual_on, style_on, concatenation = (
        helpers.resolve_pretrained_model(
            pretrained_model=[os.fspath(model_file)],
            model_type=None,
            net_avg=True,
            use_torch=True,
            model_names=["cyto"],
            bd_model_names=[],
            c2_model_names=[],
            omni=False,
        )
    )

    assert pretrained == [os.fspath(model_file)]
    assert model_name is None
    assert net_avg is True
    assert updates == {}
    assert residual_on is None
    assert style_on is None
    assert concatenation is None
