from __future__ import annotations

import logging
import os
import sys

import numpy as np
from tqdm import tqdm

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


def _resolve_model(args, device):
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
    )


def _run_evaluation(args) -> None:
    if args.tyx is not None:
        args.tyx = tuple(int(s) for s in args.tyx.split(","))

    if args.check_mkl:
        mkl_enabled = models.check_mkl((not args.mxnet))
    else:
        mkl_enabled = True

    if mkl_enabled and args.mkldnn:
        os.environ["MXNET_SUBGRAPH_BACKEND"] = "MKLDNN"
    else:
        os.environ["MXNET_SUBGRAPH_BACKEND"] = ""

    if not args.testing:
        saving_something = (
            args.save_png
            or args.save_tif
            or args.save_flows
            or args.save_ncolor
            or args.save_txt
        )
        if not saving_something and not confirm_prompt("Proceed without saving any outputs?"):
            sys.exit(0)

    device, _gpu_available = models.assign_device(args.use_gpu, args.gpu_number)
    model = _resolve_model(args, device)
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
            rescale=args.rescale,
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
            masks = utils.remove_edge_masks(masks)

        if not args.no_npy:
            io.masks_flows_to_seg(image, masks, flows, diameter, image_name, channels)

        if (
            args.save_png
            or args.save_tif
            or args.save_flows
            or args.save_ncolor
            or args.save_txt
        ):
            io.save_masks(
                image,
                masks,
                flows,
                image_name,
                png=args.save_png,
                tif=args.save_tif,
                save_flows=args.save_flows,
                save_outlines=args.save_outlines,
                save_ncolor=args.save_ncolor,
                dir_above=args.dir_above,
                savedir=args.savedir,
                save_txt=args.save_txt,
                in_folders=args.in_folders,
                omni=args.omni,
                channel_axis=args.channel_axis,
                channels=channels,
            )


def _run_training(args) -> None:
    if args.tyx is not None:
        args.tyx = tuple(int(s) for s in args.tyx.split(","))
    if args.train_size:
        raise NotImplementedError("Size model training is not implemented in omnirefactor.")
    if args.unet:
        raise NotImplementedError("UNet training is not implemented in omnirefactor.")

    device, _gpu_available = models.assign_device(args.use_gpu, args.gpu_number)

    test_dir = args.test_dir if len(args.test_dir) > 0 else None
    output = io.load_train_test_data(
        args.dir,
        test_dir=test_dir,
        image_filter=args.img_filter,
        mask_filter=args.mask_filter,
        unet=args.unet,
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

    model = _resolve_model(args, device)
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
        rescale=(args.diameter != 0),
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        dataloader=args.dataloader,
        num_workers=args.num_workers,
        min_train_masks=args.min_train_masks if args.logits else 0,
        SGD=(not args.RAdam),
        tyx=args.tyx,
        timing=args.timing,
        do_autocast=args.amp,
        affinity_field=args.affinity_field,
    )
    model.pretrained_model = cpmodel_path
    if save_path is None:
        logger.info("Model trained without saving checkpoints")
    else:
        logger.info("Model trained and saved to %s", cpmodel_path)


def main(argv: list[str] | None = None) -> None:
    args = get_arg_parser().parse_args(argv)
    if args.train or args.train_size:
        _run_training(args)
    else:
        _run_evaluation(args)
