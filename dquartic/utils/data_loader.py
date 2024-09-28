from typing import Literal
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class DIAMSDataset(Dataset):
    """
    A dataset class for loading DIA-MS/MS data.
    Args:
        ms2_file (str): The file path to the MS2 data file.
        ms1_file (str): The file path to the MS1 data file.
    Attributes:
        ms2_data (ndarray): The loaded MS2 data.
        ms1_data (ndarray): The loaded MS1 data.
    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the MS2 and MS1 samples at the given index.
    """
    def __init__(self, ms2_file, ms1_file, normalize: Literal[None, 'minmax'] = None):
        self.ms2_data = np.load(ms2_file)
        self.ms1_data = np.load(ms1_file)
        
        # Ensure the data is in float32 format
        self.ms2_data = self.ms2_data.astype(np.float32)
        self.ms1_data = self.ms1_data.astype(np.float32)
        
        # Normalize the data
        if normalize == 'minmax':
            self.ms2_data = (self.ms2_data - self.ms2_data.min()) / (self.ms2_data.max() - self.ms2_data.min())
            self.ms1_data = (self.ms1_data - self.ms1_data.min()) / (self.ms1_data.max() - self.ms1_data.min())

    def __len__(self):
        return len(self.ms2_data)

    def __getitem__(self, idx):
        ms2_sample = torch.from_numpy(self.ms2_data[idx])
        ms1_sample = torch.from_numpy(self.ms1_data[idx])
        return ms2_sample, ms1_sample

