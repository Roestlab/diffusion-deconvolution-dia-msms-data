from dquartic.model.model import DDIMDiffusionModel
from dquartic.model.building_blocks import CustomTransformer
from dquartic.utils.data_loader import DIAMSDataset

import torch
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd


def main():
    device="cuda:0"
    # Initialize model
    model = DDIMDiffusionModel(model=CustomTransformer(), num_timesteps=1000, device=device)
    model.model.load_state_dict(torch.load("best_model_global_minmax.pth"), weights_only=True)

    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    model.model.eval()
    # Create the DataLoader
    batch_size = 1  # Adjust this based on your model and memory constraints
    dataset = DIAMSDataset('ms2_data_cat_int32.npy', 'ms1_data_int32.npy', split=True)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Iterate over the DataLoader
    for ms2_1, ms1_1, ms2_2, ms1_2 in data_loader:
        x_start, x_cond = ms2_1, ms1_1  # Unpack your batch data
        x_start = x_start.to(device)
        x_cond = x_cond.to(device)
        x_start = torch.randn_like(x_start)
        sample = model.sample(x_start, x_cond, num_steps=100, eta=0.0)
        sample = sample[0].cpu().detach().numpy()
        # Get the dimensions of the array
        rows, cols = sample.shape

        # Create meshgrid for x and y coordinates
        y, x = np.meshgrid(range(rows), range(cols), indexing='ij')

        # Flatten the arrays
        x_flat = x.flatten()
        y_flat = y.flatten()
        intensity_flat = sample.flatten()

        # Create a dataframe
        df = pd.DataFrame({
            'x': x_flat,
            'y': y_flat,
            'intensity': intensity_flat
        })
        df.plot(
            x='y', y='x', z='intensity',
            kind='peakmap', xlabel="X Index", ylabel="Y Index",
            height=500, width=800, plot_3d=True,
            grid=False, show_plot=True
        )
