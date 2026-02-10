"""
Train a diffusion model on images.
"""
import argparse
import json
from diffusion_model import dist_util, logger
from diffusion_model.image_datasets import load_data_cond
from diffusion_model.resample import create_named_schedule_sampler
from diffusion_model.script_util import (
    model_and_diffusion_defaults3d,
    create_model_and_diffusion_cond3d,
    args_to_dict,
    add_dict_to_argparser,
)
from diffusion_model.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir="./log/")

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion_cond3d(
        **args_to_dict(args, model_and_diffusion_defaults3d().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data_cond(
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        crop_size=args.crop_size,
        crop_spacing=args.crop_spacing
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        model3d=True
    ).run_loop()


def create_argparser():
        
    config_path = "image_generation_scripts/config/config.json"

    defaults = json.load(open(config_path, 'r'))

    defaults.update(model_and_diffusion_defaults3d())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
