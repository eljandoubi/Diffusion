# Diffusion

A PyTorch implementation of Denoising Diffusion Probabilistic Models (DDPM) trained on the CelebA dataset. This project builds a diffusion model from scratch, including the UNet-based model architecture, the noise scheduler, and the training loop using Hugging Face Accelerate.

## Features

- **DDPM Implementation**: Complete implementation of the forward (noise injection) and reverse (denoising) diffusion processes.
- **UNet Architecture**: Custom UNet model with Self-Attention blocks and residual connections.
- **Accelerate Integration**: Leveraging Hugging Face Accelerate for easy distributed training and mixed precision support.
- **CelebA Dataset**: Automatically downloads and prepares the CelebA dataset for training.
- **TensorBoard Logging**: Tracks training progress and visualizes generated samples.

## Project Structure

```
src/
├── model.py      # UNet model definition with Attention mechanisms
├── sampler.py    # Diffusion process logic (adding/removing noise)
├── train.py      # Main training script
└── utils.py      # Utility functions for visualization and logging
```

## Installation

1. Clone the repository and navigate to the project directory.

2. Create a virtual environment and install the required dependencies:
   ```bash
   uv sync
   ```

## Usage

### Training

To start training the model, use the `src/train.py` script. The script handles dataset downloading (CelebA) automatically.

```bash
uv run src/train.py --experiment_name "my_first_diffusion" --path_to_data "./data"
```

### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--experiment_name` | Name of the experiment (folder for logs/checkpoints) | `base` |
| `--path_to_data` | Path where data will be downloaded/stored | `path/to/data` |
| `--num_training_steps` | Total number of training steps | `75000` |
| `--batch_size` | Effective batch size per GPU | `256` |
| `--learning_rate` | Max learning rate | `1e-4` |
| `--img_size` | Image resolution (height/width) | `192` |
| `--save_every` | Step interval for saving checkpoints | `5000` |

For a full list of arguments, run:
```bash
uv run src/train.py --help
```

### Monitoring

Training progress and generated samples can be monitored using TensorBoard:

```bash
tensorboard --logdir ./my_first_diffusion
```

## References

- [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)

