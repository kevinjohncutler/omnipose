import os
import pathlib

import numpy as np

from .logging import models_logger
from .. import io

_MODEL_URL = 'https://www.cellpose.org/models'
_MODEL_URL = 'https://raw.githubusercontent.com/kevinjohncutler/omnipose-models/main'
_MODEL_DIR_ENV = os.environ.get("CELLPOSE_LOCAL_MODELS_PATH")
_MODEL_DIR_DEFAULT = pathlib.Path.home().joinpath('.cellpose', 'models')
MODEL_DIR = pathlib.Path(_MODEL_DIR_ENV) if _MODEL_DIR_ENV else _MODEL_DIR_DEFAULT


def model_path(model_type, model_index, use_torch):
    torch_str = 'torch' if use_torch else ''
    basename = '%s%s_%d' % (model_type, torch_str, model_index)
    return cache_model_path(basename)



def cache_model_path(basename):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    url = f'{_MODEL_URL}/{basename}'
    cached_file = os.fspath(MODEL_DIR.joinpath(basename))
    if not os.path.exists(cached_file):
        models_logger.info('Downloading: "{}" to {}\n'.format(url, cached_file))
        print(url, cached_file)
        io.download_url_to_file(url, cached_file, progress=True)
    return cached_file


def resolve_model_init_config(
    pretrained_model,
    model_type,
    net_avg,
    model_names,
    bd_model_names,
    c2_model_names,
    omni,
    pretrained_model_exists=None,
):
    """Resolve model defaults without touching the filesystem.

    Returns
    -------
    dict
        Keys: ``model_name``, ``net_avg``, ``updates``,
        ``residual_on``/``style_on``/``concatenation``,
        ``model_indices``, ``warn_bad_path``.
    """
    model_name = None
    updates = {}
    residual_on = style_on = concatenation = None
    warn_bad_path = False

    if pretrained_model_exists is None and pretrained_model:
        # leave as None to avoid filesystem checks in pure config usage
        pretrained_model_exists = None

    if model_type is not None or (pretrained_model and pretrained_model_exists is False):
        model_name = model_type
        if not np.any([model_name == s for s in model_names]):
            model_name = "cyto"
        if pretrained_model and pretrained_model_exists is False:
            warn_bad_path = True

        nuclear = "nuclei" in model_name
        bacterial = ("bact" in model_name) or ("worm" in model_name)
        plant = "plant" in model_name

        if nuclear:
            updates["diam_mean"] = 17.0
        elif bacterial or plant:
            net_avg = False

        if model_name in bd_model_names:
            updates["nclasses"] = 3
        else:
            updates["nclasses"] = 2

        if model_name in c2_model_names:
            updates["nchan"] = 2

        if omni:
            net_avg = False

        model_indices = list(range(4)) if net_avg else [0]
        residual_on, style_on, concatenation = True, True, False
    else:
        model_indices = []

    return {
        "model_name": model_name,
        "net_avg": net_avg,
        "updates": updates,
        "residual_on": residual_on,
        "style_on": style_on,
        "concatenation": concatenation,
        "model_indices": model_indices,
        "warn_bad_path": warn_bad_path,
    }


def resolve_pretrained_model(
    pretrained_model,
    model_type,
    net_avg,
    use_torch,
    model_names,
    bd_model_names,
    c2_model_names,
    omni,
):
    """
    Resolve a model name or explicit path into a pretrained model list + model-specific defaults.
    Returns (pretrained_model, pretrained_model_string, net_avg, updates, residual_on, style_on, concatenation).
    """
    pretrained_model_exists = None
    if pretrained_model:
        pretrained_model_exists = os.path.exists(pretrained_model[0])

    config = resolve_model_init_config(
        pretrained_model=pretrained_model,
        model_type=model_type,
        net_avg=net_avg,
        model_names=model_names,
        bd_model_names=bd_model_names,
        c2_model_names=c2_model_names,
        omni=omni,
        pretrained_model_exists=pretrained_model_exists,
    )

    if config["warn_bad_path"]:
        models_logger.warning("pretrained model has incorrect path")
    if config["model_name"] is not None:
        models_logger.info(f">>{config['model_name']}<< model set to be used")
        pretrained_model = [
            model_path(config["model_name"], j, use_torch) for j in config["model_indices"]
        ]

    return (
        pretrained_model,
        config["model_name"],
        config["net_avg"],
        config["updates"],
        config["residual_on"],
        config["style_on"],
        config["concatenation"],
    )
