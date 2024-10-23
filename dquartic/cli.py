import click
from .utils.data_loader import DIAMSDataset
from .model.building_blocks import CustomTransformer
from .model.model import DDIMDiffusionModel
from .model.unet1d import Unet1D
from torch.utils.data import DataLoader
import torch
import wandb

@click.group()
def cli():
    pass


@cli.command()
@click.option('--epochs', default=1000, help='Number of epochs to train')
@click.option('--warmup-epochs', default=5, help='Number of warmup epochs for learning rate scheduler')
@click.option('--batch-size', default=1, help='Batch size for training')
@click.option('--learning-rate', default=1e-6, help='Learning rate for optimizer')
@click.option('--hidden-dim', default=1024, help='Hidden dimension for the model')
@click.option('--num-heads', default=8, help='Number of attention heads')
@click.option('--num-layers', default=8, help='Number of transformer layers')
@click.option('--num-timesteps', default=1000, help='Number of timesteps for diffusion model')
@click.option('--beta-start', default=0.001, help='Start value for beta scheduler')
@click.option('--beta-end', default=0.00125, help='End value for beta scheduler')
@click.option('--ms1-loss-weight', default=0.0, help='Weight for MS1 loss')
@click.option("--use-model", default="CustomTransformer", help="Model class to use. (CustomTransformer, Unet1D)")
# Data settings
@click.option('--normalize', default=None, help='Normalization method. (None, minmax)')
@click.option('--ms2-data-path', default='bigdata/ms2_data_cat_int32.npy', help='Path to MS2 data')
@click.option('--ms1-data-path', default='bigdata/ms1_data_int32.npy', help='Path to MS1 data')
@click.option('--checkpoint-path', default='best_model.ckpt', help='Path to save the best model')
@click.option('--use-wandb', is_flag=True, help='Enable Weights & Biases logging')
@click.option('--threads', default=4, help='Number of threads for data loading')
# Wandb settings
@click.option('--wandb-project', default='dquartic', help='Weigths & Biases project name')
@click.option('--wandb-name', default=None, help='Weigths & Biases name. If None, Wandb will generate a random name')
@click.option('--wandb-id', default=None, help='Weigths & Biases run ID')
@click.option('--wandb-resume', default=None, help='Weigths & Biases resume ID if run crashed or stopped early. Allowed values are None, allow, must, never')
@click.option('--wandb-architecture', default='DDIM(CustomTransformer)', help='Weigths & Biases model architecture name')
@click.option('--wandb-dataset', default='Josh_GPF_DIA', help='Weigths & Biases dataset name')
@click.option('--wandb-mode', default='offline', help='Weigths & Biases mode. (offline, online). Default is offline, which means metrics are only saved locally. You will need to run wandb sync to upload the metrics to the cloud.')
def train(
    epochs,
    warmup_epochs,
    batch_size,
    learning_rate,
    hidden_dim,
    num_heads,
    num_layers,
    num_timesteps,
    beta_start,
    beta_end,
    ms1_loss_weight,
    use_model,
    normalize,
    ms2_data_path,
    ms1_data_path,
    checkpoint_path,
    use_wandb,
    threads,
    wandb_project,
    wandb_name,
    wandb_id,
    wandb_resume,
    wandb_architecture,
    wandb_dataset,
    wandb_mode,
):
    """
    Train a DDIM model on the DIAMS dataset.
    """
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"Total Memory: {torch.cuda.get_device_properties(i).total_memory / (1024 ** 2)} MB")
            print(f"Allocated Memory: {torch.cuda.memory_allocated(i) / (1024 ** 2)} MB")
            print(f"Cached Memory: {torch.cuda.memory_reserved(i) / (1024 ** 2)} MB")
    else:
        print("No GPUs available.")

    dataset = DIAMSDataset(ms2_data_path, ms1_data_path, normalize=normalize)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=threads)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if use_model == "Unet1D":
        model_init = {
            "dim": 4,
            "channels" : 1,
            "dim_mults": (1, 2, 2, 3, 3, 4, 4),
            "cond_channels": 1,
            "cond_init_dim": 4,
            "has_condition": True
        }
        model = Unet1D(
            **model_init
        ).to(device)
    elif use_model == "CustomTransformer":
        model_init = {
            "input_dim": 40000,
            "hidden_dim": hidden_dim,
            "num_heads": num_heads,
            "num_layers": num_layers
        }
        model = CustomTransformer(**model_init).to(device)
    else:
        raise ValueError(f"Invalid model class: {use_model}")
    diffusion_model = DDIMDiffusionModel(model_class=model, num_timesteps=num_timesteps, beta_start=beta_start, beta_end=beta_end, ms1_loss_weight=ms1_loss_weight, device=device)

    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=wandb_name,
            id=wandb_id,
            resume=wandb_resume,
            config={
                "learning_rate": learning_rate,
                "architecture": wandb_architecture,
                "dataset": wandb_dataset,
                "epochs": epochs,
                "batch_size": batch_size,
                "beta_start": beta_start,
                "beta_end": beta_end,
                "ms1_loss_weight": ms1_loss_weight,
                "model": use_model,
                "model_params": vars(model),
                **model_init
            },
            settings=wandb.Settings(start_method="fork"),
            mode=wandb_mode,
        )

    diffusion_model.train(data_loader, batch_size, epochs, warmup_epochs, learning_rate, use_wandb, checkpoint_path)

    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    cli()
