from __future__ import annotations

import logging
import warnings

import numpy as np

from .axes import update_axis
from .imports import normalize99, move_axis, move_min_dim

transforms_logger = logging.getLogger(__name__)


def convert_image(
    x,
    channels,
    channel_axis=None,
    z_axis=None,
    do_3D=False,
    normalize=True,
    invert=False,
    nchan=2,
    dim=2,
    omni=False,
):
    """Return image with z first, channels last, and normalized intensities."""
    if x.ndim > 3:
        to_squeeze = np.array([int(isq) for isq, s in enumerate(x.shape) if s == 1])
        if len(to_squeeze) > 0:
            channel_axis = update_axis(channel_axis, to_squeeze, x.ndim) if channel_axis is not None else channel_axis
            z_axis = update_axis(z_axis, to_squeeze, x.ndim) if z_axis is not None else z_axis
        x = x.squeeze()
    if z_axis is not None and x.ndim > 2:
        x = move_axis(x, axis=z_axis, pos="first")
        if channel_axis is not None:
            channel_axis += 1
        if x.ndim == 3:
            x = x[..., np.newaxis]
    if channel_axis is not None and x.ndim > 2:
        x = move_axis(x, axis=channel_axis, pos="last")
    elif x.ndim == dim:
        x = x[np.newaxis]

    if do_3D:
        if x.ndim < 3:
            transforms_logger.critical("ERROR: cannot process 2D images in 3D mode")
            raise ValueError("ERROR: cannot process 2D images in 3D mode")
        if x.ndim < 4:
            x = x[..., np.newaxis]

    if channel_axis is None:
        x = move_min_dim(x)
        channel_axis = -1

    if x.ndim > 3:
        transforms_logger.info(
            "multi-stack tiff read in as having %d planes %d channels", x.shape[0], x.shape[-1]
        )

    if channels is not None:
        channels = channels[0] if len(channels) == 1 else channels
        if len(channels) < 2:
            transforms_logger.critical("ERROR: two channels not specified")
            raise ValueError("ERROR: two channels not specified")
        x = reshape(x, channels=channels, channel_axis=channel_axis)
    else:
        if x.shape[-1] > nchan and x.ndim > dim:
            transforms_logger.warning(
                'WARNING: more than %d channels given, use "channels" input for specifying channels -'
                "just using first %d channels to run processing",
                nchan,
                nchan,
            )
            x = x[..., :nchan]

        if not do_3D and x.ndim > 3 and dim == 2:
            transforms_logger.critical("ERROR: cannot process 4D images in 2D mode")
            raise ValueError("ERROR: cannot process 4D images in 2D mode")

        if x.shape[-1] < nchan:
            x = np.concatenate((x, np.tile(np.zeros_like(x), (1, 1, nchan - 1))), axis=-1)

    if normalize or invert:
        x = normalize_img(x, invert=invert, omni=omni)

    return x


def reshape(data, channels=(0, 0), chan_first=False, channel_axis=0):
    """Reshape data using channels."""
    data = data.astype(np.float32)
    if data.ndim < 3:
        data = data[..., np.newaxis]
    elif data.shape[0] < 8 and data.ndim == 3:
        data = np.transpose(data, (1, 2, 0))
        channel_axis = -1
    if data.shape[-1] == 1:
        data = np.concatenate((data, np.zeros_like(data)), axis=-1)
    else:
        if channels[0] == 0:
            data = data.mean(axis=channel_axis, keepdims=True)
            data = np.concatenate((data, np.zeros_like(data)), axis=-1)
        else:
            chanid = [channels[0] - 1]
            if channels[1] > 0:
                chanid.append(channels[1] - 1)
            data = data[..., chanid]
            for i in range(data.shape[-1]):
                if np.ptp(data[..., i]) == 0.0:
                    if i == 0:
                        warnings.warn("chan to seg' has value range of ZERO")
                    else:
                        warnings.warn("'chan2 (opt)' has value range of ZERO, can instead set chan2 to 0")
            if data.shape[-1] == 1:
                data = np.concatenate((data, np.zeros_like(data)), axis=-1)
    if chan_first:
        if data.ndim == 4:
            data = np.transpose(data, (3, 0, 1, 2))
        else:
            data = np.transpose(data, (2, 0, 1))
    return data


def normalize_img(img, axis=-1, invert=False, omni=False):
    """Normalize each channel to [0,1] using 1st/99th percentiles."""
    if img.ndim < 3:
        error_message = "Image needs to have at least 3 dimensions"
        transforms_logger.critical(error_message)
        raise ValueError(error_message)

    img = img.astype(np.float32)
    img = np.moveaxis(img, axis, 0)
    for k in range(img.shape[0]):
        if np.percentile(img[k], 99) > np.percentile(img[k], 1) + 1e-3:
            img[k] = normalize99(img[k], omni=omni)
            if invert:
                img[k] = -1 * img[k] + 1
    img = np.moveaxis(img, 0, axis)
    return img


def reshape_train_test(train_data, train_labels, test_data, test_labels, channels, channel_axis=0, normalize=True, dim=2, omni=False):
    nimg = len(train_data)
    if nimg != len(train_labels):
        error_message = "train data and labels not same length"
        transforms_logger.critical(error_message)
        raise ValueError(error_message)

    if train_labels[0].ndim < 2 or train_data[0].ndim < 2:
        error_message = "training data or labels are not at least two-dimensional"
        transforms_logger.critical(error_message)
        raise ValueError(error_message)

    if train_data[0].ndim > 3:
        error_message = "training data is more than three-dimensional (should be 2D or 3D array)"
        transforms_logger.critical(error_message)
        raise ValueError(error_message)

    if not (test_data is not None and test_labels is not None and len(test_data) > 0 and len(test_data) == len(test_labels)):
        test_data = None

    train_data, test_data, run_test = reshape_and_normalize_data(
        train_data,
        test_data=test_data,
        channels=channels,
        channel_axis=channel_axis,
        normalize=normalize,
        omni=omni,
        dim=dim,
    )

    if train_data is None:
        error_message = "training data do not all have the same number of channels"
        transforms_logger.critical(error_message)
        raise ValueError(error_message)

    if not run_test:
        test_data, test_labels = None, None

    if not np.all([dta.shape[-dim:] == lbl.shape[-dim:] for dta, lbl in zip(train_data, train_labels)]):
        error_message = "training data and labels are not the same shape, must be something wrong with preprocessing assumptions"
        transforms_logger.critical(error_message)
        raise ValueError(error_message)

    return train_data, train_labels, test_data, test_labels, run_test


def reshape_and_normalize_data(train_data, test_data=None, channels=None, channel_axis=0, normalize=True, omni=False, dim=2):
    for test, data in enumerate([train_data, test_data]):
        if data is None:
            return train_data, test_data, False
        nimg = len(data)
        for i in range(nimg):
            if channels is None:
                if channel_axis is not None:
                    data[i] = move_axis(data[i], axis=channel_axis, pos="first")
                else:
                    transforms_logger.warning(
                        "No channel axis specified. Image shape is %s. Supply channel_axis if incorrect.",
                        data[i].shape,
                    )

            if channels is not None:
                data[i] = reshape(data[i], channels=channels, chan_first=True, channel_axis=channel_axis)

            if channels is None and data[i].ndim == dim:
                data[i] = data[i][np.newaxis]

            if normalize:
                data[i] = normalize_img(data[i], axis=0, omni=omni)

    return train_data, test_data, True

