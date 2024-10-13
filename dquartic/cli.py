import click
from .utils.data_loader import DIAMSDataset
from .model.building_blocks import CustomTransformer
from .model.model import DDIMDiffusionModel, train_model
from torch.utils.data import DataLoader
import torch
import wandb

@click.group()
def cli():
    pass

@cli.command()
@click.option('--epochs', default=1000, help='Number of epochs to train')
@click.option('--batch-size', default=1, help='Batch size for training')
@click.option('--learning-rate', default=1e-6, help='Learning rate for optimizer')
@click.option('--hidden-dim', default=1024, help='Hidden dimension for the model')
@click.option('--num-heads', default=8, help='Number of attention heads')
@click.option('--num-layers', default=8, help='Number of transformer layers')
@click.option('--normalize', default=None, help='Normalization method. (None, minmax)')
@click.option('--ms2-data-path', default='bigdata/ms2_data_cat_int32.npy', help='Path to MS2 data')
@click.option('--ms1-data-path', default='bigdata/ms1_data_int32.npy', help='Path to MS1 data')
@click.option('--checkpoint-path', default='best_model.pth', help='Path to save the best model')
@click.option('--use-wandb', is_flag=True, help='Enable Weights & Biases logging')
@click.option('--threads', default=4, help='Number of threads for data loading')
@click.option('--use-checkpoint', default=False, help='Continue training from a previous checkpoint saved at checkpoint-path')
def train(epochs, batch_size, learning_rate, hidden_dim, num_heads, num_layers, split, normalize, ms2_data_path, ms1_data_path, checkpoint_path, use_wandb, threads, use_checkpoint):
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
        
    # Your training code here
    dataset = DIAMSDataset(ms2_data_path, ms1_data_path, normalize=normalize)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=threads)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    model = CustomTransformer(input_dim=40000, hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers).to(device)
    diffusion_model = DDIMDiffusionModel(model=model, num_timesteps=1000, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    if use_checkpoint:
        try:
            diffusion_model.load(checkpoint_path) 
        except Exception as e:
            print(f"Error loading from checkpoint: {e}")
    if use_wandb:
        wandb.init(project="dquartic", config={
            "learning_rate": learning_rate,
            "architecture": "DDIM",
            "dataset": "Josh_GPF_DIA",
            "epochs": epochs,
            "batch_size": batch_size,
            "hidden_dim": hidden_dim,
            "num_heads": num_heads,
            "num_layers": num_layers
        })

    train_model(diffusion_model, data_loader, optimizer, epochs, device, use_wandb, checkpoint_path, num_warmup_steps, num_training_steps)


    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    cli()
