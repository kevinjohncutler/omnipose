from __future__ import annotations

import logging
import os
import sys
import random

import numpy as np
from tqdm import tqdm
import torch

from .. import io, models, utils
from ..logger import TqdmToLogger
from .parser import get_arg_parser

logger = logging.getLogger(__name__)


def confirm_prompt(question: str) -> bool:
    reply = None
    while reply not in ("", "y", "n"):
        reply = input(f"{question} (y/n): ").lower()
    return reply in ("", "y")


def _resolve_channels(args):
    if args.nchan == 2 and not args.all_channels:
        return [args.chan, args.chan2]
    return None


def _resolve_model(args, device, norm_type: str):
    builtin_model = np.any([args.pretrained_model == s for s in models.MODEL_NAMES])
    if args.pretrained_model is not None and "omni" in args.pretrained_model:
        args.omni = True

    if builtin_model:
        return models.OmniModel(
            gpu=args.use_gpu,
            device=device,
            model_type=args.pretrained_model,
            use_torch=(not args.mxnet),
            omni=args.omni,
            net_avg=(not args.fast_mode and not args.no_net_avg),
            nclasses=args.nclasses,
            logits=args.logits,
            nsample=args.nsample,
            nchan=args.nchan,
            dim=args.dim,
            norm_type=norm_type,
        )

    if args.pretrained_model is not None and not os.path.exists(args.pretrained_model):
        logger.warning("Provided model path not found, defaulting to cyto model")
        args.pretrained_model = "cyto"
        return models.OmniModel(
            gpu=args.use_gpu,
            device=device,
            model_type=args.pretrained_model,
            use_torch=True,
            omni=args.omni,
            net_avg=False,
            nclasses=args.nclasses,
            logits=args.logits,
            nsample=args.nsample,
            nchan=args.nchan,
            dim=args.dim,
            norm_type=norm_type,
        )

    return models.OmniModel(
        gpu=args.use_gpu,
        device=device,
        pretrained_model=args.pretrained_model,
        use_torch=True,
        omni=args.omni,
        net_avg=False,
        nclasses=args.nclasses,
        logits=args.logits,
        nsample=args.nsample,
        nchan=args.nchan,
        dim=args.dim,
        norm_type=norm_type,
    )


def _run_evaluation(args) -> None:
    if args.tyx is not None:
        args.tyx = tuple(int(s) for s in args.tyx.split(","))

    os.environ["MXNET_SUBGRAPH_BACKEND"] = ""

    if not args.testing:
        saving_something = (
            args.save_png
            or args.save_tif
            or args.save_flows
            or args.save_ncolor
        )
        if not saving_something and not confirm_prompt("Proceed without saving any outputs?"):
            sys.exit(0)

    device, _gpu_available = models.assign_device(args.use_gpu, args.gpu_number)
    norm_type = "batch"
    model = _resolve_model(args, device, norm_type)
    channels = _resolve_channels(args)

    image_names = io.get_image_files(
        args.dir,
        args.mask_filter,
        img_filter=args.img_filter,
        look_one_level_down=args.look_one_level_down,
    )
    if not image_names:
        raise ValueError(f"No images found under {args.dir}")

    cellpose_model = np.any([args.pretrained_model == s for s in models.CP_MODELS])
    if args.diameter == 0:
        diameter = None if cellpose_model else model.diam_mean
        logger.info("Estimating diameter for each image" if diameter is None else "Using model diam_mean")
    else:
        diameter = args.diameter
        logger.info("Using diameter %s for all images", diameter)

    tqdm_out = TqdmToLogger(logger, level=logging.INFO)
    for image_name in tqdm(image_names, file=tqdm_out):
        image = io.imread(image_name)
        masks, flows, _ = model.eval(
            image,
            channels=channels,
            diameter=diameter,
            rescale_factor=args.rescale,
            do_3D=args.do_3D,
            net_avg=(not args.fast_mode and not args.no_net_avg),
            augment=False,
            resample=(not args.no_resample and not args.fast_mode),
            flow_threshold=args.flow_threshold,
            mask_threshold=args.mask_threshold,
            niter=args.niter,
            diam_threshold=args.diam_threshold,
            invert=args.invert,
            batch_size=args.batch_size,
            interp=(not args.no_interp),
            cluster=args.cluster,
            suppress=(not args.no_suppress),
            channel_axis=args.channel_axis,
            z_axis=args.z_axis,
            omni=args.omni,
            affinity_seg=args.affinity_seg,
            anisotropy=args.anisotropy,
            verbose=args.verbose,
            min_size=args.min_size,
            max_size=args.max_size,
            transparency=args.transparency,
            model_loaded=True,
        )

        if args.exclude_on_edges:
            masks = utils.clean_boundary(
                masks,
                boundary_thickness=1,
                area_thresh=np.inf,
                cutoff=0.0,
            )
            utils.fastremap.renumber(masks, in_place=True)

        if not args.no_npy:
            io.masks_flows_to_seg(image, masks, flows, diameter, image_name, channels)

        if (
            args.save_png
            or args.save_tif
            or args.save_flows
            or args.save_ncolor
        ):
            io.save_masks(
                image,
                masks,
                flows,
                image_name,
                png=args.save_png,
                tif=args.save_tif,
                save_flows=args.save_flows,
                save_ncolor=args.save_ncolor,
                dir_above=args.dir_above,
                savedir=args.savedir,
                in_folders=args.in_folders,
                omni=args.omni,
                channel_axis=args.channel_axis,
                channels=channels,
            )


def _run_training(args) -> None:
    if args.tyx is not None:
        args.tyx = tuple(int(s) for s in args.tyx.split(","))
    if args.batch_size is not None and args.batch_size < 2:
        logger.warning("Batch size %s is too small; using 2 instead.", args.batch_size)
        args.batch_size = 2
    device, _gpu_available = models.assign_device(args.use_gpu, args.gpu_number)
    norm_type = "batch"

    test_dir = args.test_dir if len(args.test_dir) > 0 else None
    output = io.load_train_test_data(
        args.dir,
        test_dir=test_dir,
        image_filter=args.img_filter,
        mask_filter=args.mask_filter,
        look_one_level_down=args.look_one_level_down,
        omni=args.omni,
        do_links=args.links,
    )

    (
        images,
        labels,
        links,
        image_names,
        test_images,
        test_labels,
        test_links,
        image_names_test,
    ) = output

    model = _resolve_model(args, device, norm_type)
    save_path = None if args.no_save else os.path.realpath(args.dir)
    cpmodel_path = model.train(
        images,
        labels,
        links,
        train_files=image_names,
        test_data=test_images,
        test_labels=test_labels,
        test_links=test_links,
        test_files=image_names_test,
        learning_rate=args.learning_rate,
        channels=None,
        channel_axis=args.channel_axis,
        save_path=save_path,
        save_every=args.save_every,
        save_each=args.save_each,
        do_rescale=(args.diameter != 0),
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        min_train_masks=args.min_train_masks if args.logits else 0,
        SGD=(not args.RAdam),
        tyx=args.tyx,
        timing=args.timing,
        do_autocast=args.amp,
        affinity_field=args.affinity_field,
        tensorboard=args.tensorboard,
        sym_kernels=args.sym_kernels,
        symmetry_weight=args.symmetry_weight,
    )
    model.pretrained_model = cpmodel_path
    if save_path is None:
        logger.info("Model trained without saving checkpoints")
    else:
        logger.info("Model trained and saved to %s", cpmodel_path)


def main(argv: list[str] | None = None) -> None:
    args = get_arg_parser().parse_args(argv)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        from ..gpu.device import seed_all
        seed_all(args.seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        from ..gpu.device import _get_gpu_torch
        _, gpu_available = _get_gpu_torch()
        warn_only = bool(args.use_gpu and gpu_available)
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        torch.autograd.set_detect_anomaly(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.enabled = False
    if args.train:
        _run_training(args)
    else:
        _run_evaluation(args)
