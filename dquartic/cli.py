import click
import ast
from .utils.data_loader import DIAMSDataset
from .utils.config_loader import load_train_config, generate_train_config
from .model.building_blocks import CustomTransformer
from .model.model import DDIMDiffusionModel
from .model.unet1d import Unet1D
from torch.utils.data import DataLoader
import torch
import wandb


# https://stackoverflow.com/a/47730333
class PythonLiteralOption(click.Option):
    def type_cast_value(self, ctx, value):
        if not isinstance(value, str):  # required for Click>=8.0.0
            return value
        try:
            return ast.literal_eval(value)
        except Exception:
            raise click.BadParameter(value)


@click.group(chain=True)
@click.version_option()
def cli():
    """
    Diffusion Deconvolution of DIA-MS/MS Data (D^4)
    """


@cli.command()
@click.argument("config-path", type=click.Path(exists=True), required=True)
@click.option("--ms2-data-path", default=None, help="Path to MS2 data, overides config file")
@click.option("--ms1-data-path", default=None, help="Path to MS1 data, overides config file")
@click.option("--batch-size", default=None, help="Batch size for training, overides config file")
@click.option("--checkpoint-path", default=None, help="Path to save the best model, overides config file")
@click.option("--use-wandb", default=None, cls=PythonLiteralOption, help="Use wandb for logging, overides config file")
@click.option("--threads", default=None, help="Number of threads for data loading, overides config file")
def train(
    config_path,
    ms2_data_path,
    ms1_data_path,
    batch_size,
    checkpoint_path,
    use_wandb,
    threads
):
    """
    Train a DDIM model on the DIAMS dataset.
    """
    if torch.cuda.is_available():
        click.echo("--" * 30)
        click.echo("GPU Information:")
        click.echo("--" * 30)
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(
                f"Total Memory: {torch.cuda.get_device_properties(i).total_memory / (1024 ** 2)} MB"
            )
            print(f"Allocated Memory: {torch.cuda.memory_allocated(i) / (1024 ** 2)} MB")
            print(f"Cached Memory: {torch.cuda.memory_reserved(i) / (1024 ** 2)} MB")
        click.echo("--" * 30)
    else:
        click.echo("--" * 30)
        print("No GPUs available.")
        click.echo("--" * 30)
    
    click.echo(f"Info: Loading config from {config_path}")
        
    config = load_train_config(config_path, ms2_data_path=ms2_data_path, ms1_data_path=ms1_data_path, batch_size=batch_size, checkpoint_path=checkpoint_path, use_wandb=use_wandb, threads=threads)
    
    ms2_data_path = config['data']['ms2_data_path']
    ms1_data_path = config['data']['ms1_data_path']
    batch_size = config['model']['batch_size']
    checkpoint_path = config['model']['checkpoint_path']
    use_wandb = config['wandb']['use_wandb']
    threads = config['threads']

    dataset = DIAMSDataset(ms2_data_path, ms1_data_path, normalize=config['data']['normalize'])
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config['model']['use_model'] == "Unet1D":
        model_init = {
            "dim": config['model']['Unet1D']['dim'],
            "channels": config['model']['Unet1D']['channels'],
            "dim_mults": tuple(config['model']['Unet1D']['dim_mults']),
            "cond_channels": config['model']['Unet1D']['cond_channels'],
            "cond_init_dim": config['model']['Unet1D']['cond_init_dim'],
            "has_condition": config['model']['Unet1D']['has_condition'],
        }
        model = Unet1D(**model_init).to(device)
    elif config['model']['use_model'] == "CustomTransformer":
        model_init = {
            "input_dim": config['model']['CustomTransformer']['input_dim'],
            "hidden_dim": config['model']['CustomTransformer']['hidden_dim'],
            "num_heads": config['model']['CustomTransformer']['num_heads'],
            "num_layers": config['model']['CustomTransformer']['num_layers'],
        }
        model = CustomTransformer(**model_init).to(device)
    else:
        raise ValueError(f"Invalid model class: {config['model']['use_model']}")
    diffusion_model = DDIMDiffusionModel(
        model_class=model,
        num_timesteps=config['model']['num_timesteps'],
        beta_start=config['model']['beta_start'],
        beta_end=config['model']['beta_end'],
        pred_type=config['model']['pred_type'],
        auto_normalize=config['model']['auto_normalize'],
        ms1_loss_weight=config['model']['ms1_loss_weight'],
        device=device,
    )

    if use_wandb:
        wandb.init(
            project=config['wandb']['wandb_project'],
            name=config['wandb']['wandb_name'],
            id=config['wandb']['wandb_id'],
            resume=config['wandb']['wandb_resume'],
            config={
                "architecture": config['wandb']['wandb_architecture'],
                "dataset": config['wandb']['wandb_dataset'],
                **config['model']
            },
            settings=wandb.Settings(start_method="fork"),
            mode=config['wandb']['wandb_mode'],
        )

    diffusion_model.train(
        data_loader, batch_size, config['model']['num_epochs'], config['model']['warmup_epochs'], config['model']['learning_rate'], use_wandb, checkpoint_path
    )

    if use_wandb:
        wandb.finish()

@cli.command()
@click.argument("config-path", type=click.Path(exists=True), required=True)
def generate_config(config_path):
    """
    Generate a training configuration file.
    """
    click.echo(f"Info: Generating config at {config_path}")
    generate_train_config(config_path)


