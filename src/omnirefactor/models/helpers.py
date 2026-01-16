import os
import pathlib

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


def size_model_path(model_type, use_torch):
    torch_str = 'torch' if use_torch else ''
    basename = 'size_%s%s_0.npy' % (model_type, torch_str)
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


def deprecation_warning_cellprob_dist_threshold(cellprob_threshold, dist_threshold):
    models_logger.warning(
        'cellprob_threshold and dist_threshold are being deprecated in a future release, use mask_threshold instead'
    )
    return cellprob_threshold if cellprob_threshold is not None else dist_threshold
