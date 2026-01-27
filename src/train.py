import os
import shutil
import argparse
import torch
from accelerate import Accelerator
from torchvision import transforms
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from model import Diffusion
from sampler import Sampler
from utils import sample_plot_image


def trainer(args: argparse.Namespace):
    ### PREP ACCELERATOR AND TRACKERS ###
    path_to_experiment = os.path.join(args.working_directory, args.experiment_name)
    accelerator = Accelerator(
        project_dir=path_to_experiment,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="tensorboard",
    )
    accelerator.init_trackers(args.experiment_name)
    tracker = accelerator.get_tracker("tensorboard")

    ### PREP DATALOADER ###
    image2tensor = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1),
        ]
    )
    mini_batch_size = args.batch_size // args.gradient_accumulation_steps
    dataset = CelebA(
        args.path_to_data,
        download=True,
        transform=image2tensor,
        target_type="identity",
    )
    trainloader = DataLoader(
        dataset,
        batch_size=mini_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    ### DEFINE MODEL ###
    model = Diffusion(
        in_channels=3,
        start_dim=args.starting_channels,
        dim_mults=(1, 2, 3, 4),
        residual_blocks_per_group=2,
        time_embed_dim=args.starting_channels * 2,
    )

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    accelerator.print("Number of Parameters:", params)

    ### PREPARE OPTIMIZER ###
    if (not args.bias_weight_decay) or (not args.norm_weight_decay):
        accelerator.print("Disabling Weight Decay on Some Parameters")
        weight_decay_params = []
        no_weight_decay_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                ### Dont have Weight decay on any bias parameter (including norm) ###
                if "bias" in name and not args.bias_weight_decay:
                    no_weight_decay_params.append(param)

                ### Dont have Weight Decay on any Norm scales params (weights) ###
                elif "groupnorm" in name and not args.norm_weight_decay:
                    no_weight_decay_params.append(param)

                else:
                    weight_decay_params.append(param)

        optimizer_group = [
            {"params": weight_decay_params, "weight_decay": args.weight_decay},
            {"params": no_weight_decay_params, "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_group, lr=args.learning_rate)

    else:
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    ### DEFINE SCHEDULER ###
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps * accelerator.num_processes,
        num_training_steps=args.num_training_steps * accelerator.num_processes,
    )

    ### PREPARE EVERYTHING ###
    model, optimizer, trainloader, scheduler = accelerator.prepare(
        model, optimizer, trainloader, scheduler
    )
    accelerator.register_for_checkpointing(scheduler)

    ### RESUME FROM CHECKPOINT ###
    if args.resume_from_checkpoint is not None:
        path_to_checkpoint = os.path.join(
            path_to_experiment, args.resume_from_checkpoint
        )
        accelerator.load_state(path_to_checkpoint)
        completed_steps = int(args.resume_from_checkpoint.split("_")[-1])
        accelerator.print(f"Resuming from Iteration: {completed_steps}")
    else:
        completed_steps = 0

    ### DEFINE DDPM SAMPLER ###
    ddpm_sampler = Sampler(
        total_timesteps=args.num_diffusion_timesteps, device=accelerator.device
    )

    ### DEFINE LOSS FN ###
    loss_functions = {"mse": nn.MSELoss(), "mae": nn.L1Loss(), "huber": nn.HuberLoss()}
    loss_fn = loss_functions[args.loss_fn]

    ### DEFINE TRAINING LOOP ###
    progress_bar = tqdm(
        range(completed_steps, args.num_training_steps),
        disable=not accelerator.is_main_process,
    )
    accumulated_loss = 0
    train = True

    while train:
        for images, ids in trainloader:
            del ids

            with accelerator.accumulate(model):
                ### Grab Number of Samples in Batch ###
                batch_size = images.shape[0]

                ### Random Sample T ###
                timesteps = torch.randint(
                    0,
                    args.num_diffusion_timesteps,
                    (batch_size,),
                    device=accelerator.device,
                )

                ### Get Noisy Images ###
                noisy_images, noise = ddpm_sampler.add_noise(images, timesteps)

                ### Get Noise Prediction ###
                noise_pred = model(
                    noisy_images,
                    timesteps,
                )

                ### Compute Error ###
                loss = loss_fn(noise_pred, noise)
                accumulated_loss += loss / args.gradient_accumulation_steps

                ### Compute Gradients ###
                accelerator.backward(loss)

                ### Clip Gradients ###
                if accelerator.sync_gradients:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                loss_gathered = accelerator.gather_for_metrics(accumulated_loss)
                mean_loss_gathered = torch.mean(loss_gathered).item()

                accelerator.log(
                    {
                        "loss": mean_loss_gathered,
                        "learning_rate": scheduler.get_last_lr()[0],
                        "iteration": completed_steps,
                    },
                    step=completed_steps,
                )

                ### Reset and Iterate ###
                accumulated_loss = 0
                completed_steps += 1
                progress_bar.update(1)

                ### EVALUATION CHECK ###
                if completed_steps % args.evaluation_interval == 0:
                    ### Save Checkpoint ###
                    path_to_checkpoint = os.path.join(
                        path_to_experiment, f"checkpoint_{completed_steps}"
                    )
                    accelerator.save_state(output_dir=path_to_checkpoint)

                    ### Delete Old Checkpoints ###
                    if accelerator.is_main_process:
                        all_checkpoints = os.listdir(path_to_experiment)
                        all_checkpoints = sorted(
                            all_checkpoints,
                            key=lambda x: int(x.split(".")[0].split("_")[-1]),
                        )

                        if len(all_checkpoints) > args.num_keep_checkpoints:
                            checkpoints_to_delete = all_checkpoints[
                                : -args.num_keep_checkpoints
                            ]

                            for checkpoint_to_delete in checkpoints_to_delete:
                                path_to_checkpoint_to_delete = os.path.join(
                                    path_to_experiment, checkpoint_to_delete
                                )
                                if os.path.isdir(path_to_checkpoint_to_delete):
                                    shutil.rmtree(path_to_checkpoint_to_delete)

                    ### Inference Model and Save Results ###
                    accelerator.print("Generating Images")
                    if accelerator.is_main_process:
                        fig = sample_plot_image(
                            step_idx=completed_steps,
                            total_timesteps=args.num_diffusion_timesteps,
                            sampler=ddpm_sampler,
                            image_size=args.img_size,
                            num_channels=3,
                            plot_freq=args.plot_freq_interval,
                            model=model,
                            num_gens=args.num_generations,
                            path_to_generated_dir=args.generated_directory,
                        )
                        tracker.log({"plot": fig}, step=completed_steps)

                if completed_steps >= args.num_training_steps:
                    train = False
                    accelerator.print("Completed Training")
                    break

    accelerator.end_training()


if __name__ == "__main__":
    #### DEFINE ARGUMENT PARSER ###
    parser = argparse.ArgumentParser(description="Arguments for Denoising Diffusion")
    parser.add_argument(
        "--experiment_name", help="Name of Training Run", required=True, type=str
    )
    parser.add_argument(
        "--path_to_data", help="Path to CelebA Dataset", default="data", type=str
    )
    parser.add_argument(
        "--working_directory",
        help="Working Directory where Checkpoints and Logs are stored",
        default=".",
        type=str,
    )
    parser.add_argument(
        "--num_keep_checkpoints",
        help="Number of the most recent checkpoints to keep",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--generated_directory",
        help="Path to folder to store all generated images during training",
        default="gen_images",
        type=str,
    )
    parser.add_argument(
        "--num_diffusion_timesteps",
        help="Number of timesteps for forward/reverse diffusion process",
        default=1000,
        type=int,
    )
    parser.add_argument(
        "--plot_freq_interval",
        help="Time pacing between generated images for reverse diffusion visuals",
        default=100,
        type=int,
    )
    parser.add_argument(
        "--num_generations",
        help="Number of generated images in each visual",
        default=5,
        type=int,
    )
    parser.add_argument(
        "--num_training_steps",
        help="Number of training steps to take",
        default=150000,
        type=int,
    )
    parser.add_argument(
        "--evaluation_interval",
        help="Number of iterations for every evaluation and plotting",
        default=5000,
        type=int,
    )
    parser.add_argument(
        "--batch_size",
        help="Effective batch size per GPU, multiplied by number of GPUs used",
        default=256,
        type=int,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        help="Number of gradient accumulation steps, splitting set batchsize",
        default=64,
        type=int,
    )
    parser.add_argument(
        "--learning_rate",
        help="Max learning rate for Cosine LR Scheduler",
        default=1e-4,
        type=float,
    )
    parser.add_argument(
        "--warmup_steps",
        help="Number of learning rate warmup steps of training",
        default=5000,
        type=int,
    )
    parser.add_argument(
        "--bias_weight_decay",
        help="Apply weight decay to bias",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--norm_weight_decay",
        help="Apply weight decay to normalization weight and bias",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--max_grad_norm",
        help="Maximum gradient norm for clipping",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--weight_decay",
        help="Weight decay constant for AdamW optimizer",
        default=1e-4,
        type=float,
    )
    parser.add_argument(
        "--loss_fn", choices=("mae", "mse", "huber"), default="mse", type=str
    )
    parser.add_argument(
        "--img_size",
        help="Width and Height of Images passed to model",
        default=192,
        type=int,
    )
    parser.add_argument(
        "--starting_channels",
        help="Number of channels in first convolutional projection",
        default=224,
        type=int,
    )
    parser.add_argument(
        "--num_workers", help="Number of workers for DataLoader", default=16, type=int
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        help="Checkpoint folder for model to resume training from, inside the experiment folder",
        default=None,
        type=str,
    )
    args = parser.parse_args()
    trainer(args)
