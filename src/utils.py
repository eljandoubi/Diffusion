import os
import itertools
import torch
import torch.nn as nn
import numpy as np
from torch.nn import Module
import matplotlib.pyplot as plt
from torchvision import transforms
from sampler import Sampler


@torch.no_grad()
def sample_plot_image(
    step_idx: int,
    total_timesteps: int,
    image_size: int,
    num_channels: int,
    plot_freq: int,
    model: Module,
    num_gens: int,
    path_to_generated_dir: str,
    sampler: Sampler,
):
    ### Conver Tensor back to Image (From Huggingface Annotated Diffusion) ###
    tensor2image_transform = transforms.Compose(
        [
            transforms.Lambda(lambda t: t.squeeze(0))
            if num_channels == 3
            else nn.Identity(),
            transforms.Lambda(lambda t: (t + 1) / 2),
            transforms.Lambda(lambda t: t.permute(1, 2, 0)),
            transforms.Lambda(lambda t: t * 255.0),
            transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
            transforms.ToPILImage(),
        ]
    )

    images = torch.randn(
        (num_gens, num_channels, image_size, image_size), device=sampler.device
    )
    num_images_per_gen = total_timesteps // plot_freq

    images_to_vis = [[] for _ in range(num_gens)]
    for t in range(total_timesteps - 1, -1, -1):
        ts = torch.full((num_gens,), t, device=sampler.device)
        noise_pred = model(images, ts)
        images = sampler.remove_noise(images, ts, noise_pred)
        if t % plot_freq == 0:
            for idx, image in enumerate(images):
                images_to_vis[idx].append(tensor2image_transform(image))

    images_to_vis = list(itertools.chain(*images_to_vis))

    fig, axes = plt.subplots(
        nrows=num_gens, ncols=num_images_per_gen, figsize=(num_images_per_gen, num_gens)
    )
    plt.tight_layout()
    for ax, image in zip(axes.ravel(), images_to_vis):
        ax.imshow(image)
        ax.axis("off")
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(os.path.join(path_to_generated_dir, f"step_{step_idx}.png"))

    return fig
