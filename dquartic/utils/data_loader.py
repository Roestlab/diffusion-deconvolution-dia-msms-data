from typing import Literal
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Literal


class DIAMSDataset(Dataset):
    """
    A dataset class for loading DIA-MS data.
    
    Args:
        ms2_file (str): The file path to the MS2 data.
        ms1_file (str): The file path to the MS1 data.
        normalize (Literal[None, 'minmax'], optional): The normalization method. Defaults to None.
        
    Attributes:
        ms2_data (numpy.ndarray): The loaded MS2 data.
        ms1_data (numpy.ndarray): The loaded MS1 data.
        normalize (Literal[None, 'minmax']): The normalization method.
        used_pairs (set): A set to keep track of used pairs.
        epoch_reset (bool): A flag to indicate if the epoch has been reset.
        
    Methods:
        __len__(): Returns the length of the dataset.
        reset_epoch(): Resets the used pairs at the start of each epoch.
        __getitem__(idx): Retrieves an item from the dataset.
    """
    def __init__(self, ms2_file, ms1_file, normalize: Literal[None, 'minmax'] = None):
        self.ms2_data = np.load(ms2_file, mmap_mode='r')
        self.ms1_data = np.load(ms1_file, mmap_mode='r')
        
        self.normalize = normalize
        self.used_pairs = set()
        self.epoch_reset = False

    def __len__(self):
        return len(self.ms2_data)

    def reset_epoch(self):
        """Reset the used pairs at the start of each epoch."""
        self.used_pairs.clear()
        self.epoch_reset = True

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset.
        Args:
            idx (int): The index of the item to retrieve.
        Returns:
            tuple: A tuple containing four torch tensors representing the MS2 and MS1 samples.
        """
        while True:
            idx_1 = random.randint(0, len(self.ms2_data) - 1)
            idx_2 = random.randint(0, len(self.ms2_data) - 1)

            if idx_1 == idx_2:
                continue

            pair = tuple(sorted((idx_1, idx_2)))

            if pair in self.used_pairs:
                continue

            self.used_pairs.add(pair)
            break

        ms2_sample_split_1 = self.ms2_data[idx_1]
        ms1_sample_split_1 = self.ms1_data[idx_1]
        ms2_sample_split_2 = self.ms2_data[idx_2]
        ms1_sample_split_2 = self.ms1_data[idx_2]
        
        if self.normalize == 'minmax':
            self.ms2_min = np.min([ms2_sample_split_1.min(), ms2_sample_split_2.min()])
            self.ms2_max = np.max([ms2_sample_split_1.max(), ms2_sample_split_2.max()])
            self.ms1_min = np.min([ms1_sample_split_1.min()])
            self.ms1_max = np.max([ms1_sample_split_1.max()])
            
            ms2_sample_split_1 = (ms2_sample_split_1 - self.ms2_min) / (self.ms2_max - self.ms2_min)
            ms1_sample_split_1 = (ms1_sample_split_1 - self.ms1_min) / (self.ms1_max - self.ms1_min)
            ms2_sample_split_2 = (ms2_sample_split_2 - self.ms2_min) / (self.ms2_max - self.ms2_min)
            ms1_sample_split_2 = (ms1_sample_split_2 - self.ms1_min) / (self.ms1_max - self.ms1_min)

        return (torch.from_numpy(ms2_sample_split_1.astype(np.float32)),
                torch.from_numpy(ms1_sample_split_1.astype(np.float32)),
                torch.from_numpy(ms2_sample_split_2.astype(np.float32)),
                torch.from_numpy(ms1_sample_split_2.astype(np.float32)))


