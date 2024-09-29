from typing import Literal
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Literal


class DIAMSDataset(Dataset):
    """
    A dataset class for loading DIA-MS/MS data with an option for random splits.
    
    Args:
        ms2_file (str): The file path to the MS2 data file.
        ms1_file (str): The file path to the MS1 data file.
        normalize (Literal[None, 'minmax']): Normalization method to apply.
        split (bool): Whether to create two random split sets.
        split_ratio (float): Ratio for the first split (default: 0.5).
    
    Attributes:
        ms2_data (ndarray): The loaded MS2 data.
        ms1_data (ndarray): The loaded MS1 data.
        split (bool): Whether the data is split.
        split_indices (list): Indices for the split if enabled.
    
    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the MS2 and MS1 samples, optionally split.
    """
    def __init__(self, ms2_file, ms1_file, normalize: Literal[None, 'minmax'] = None, split: bool = False, split_ratio: float = 0.5):
        self.ms2_data = np.load(ms2_file)
        self.ms1_data = np.load(ms1_file)
        
        # Ensure the data is in float32 format
        self.ms2_data = self.ms2_data.astype(np.float32)
        self.ms1_data = self.ms1_data.astype(np.float32)
        
        # Normalize the data
        if normalize == 'minmax':
            self.ms2_data = (self.ms2_data - self.ms2_data.min()) / (self.ms2_data.max() - self.ms2_data.min())
            self.ms1_data = (self.ms1_data - self.ms1_data.min()) / (self.ms1_data.max() - self.ms1_data.min())
        
        self.split = split
        if split:
            total_indices = list(range(len(self.ms2_data)))
            random.shuffle(total_indices)
            split_point = int(len(total_indices) * split_ratio)
            self.split_indices = [total_indices[:split_point], total_indices[split_point:]]

    def __len__(self):
        return len(self.ms2_data)

    def __getitem__(self, idx):
        if self.split:
            split_1_idx = self.split_indices[0][idx % len(self.split_indices[0])]
            split_2_idx = self.split_indices[1][idx % len(self.split_indices[1])]
            
            ms2_sample_split_1 = torch.from_numpy(self.ms2_data[split_1_idx])
            ms1_sample_split_1 = torch.from_numpy(self.ms1_data[split_1_idx])
            ms2_sample_split_2 = torch.from_numpy(self.ms2_data[split_2_idx])
            ms1_sample_split_2 = torch.from_numpy(self.ms1_data[split_2_idx])
            
            return ms2_sample_split_1, ms1_sample_split_1, ms2_sample_split_2, ms1_sample_split_2
        else:
            ms2_sample = torch.from_numpy(self.ms2_data[idx])
            ms1_sample = torch.from_numpy(self.ms1_data[idx])
            return ms2_sample, ms1_sample
