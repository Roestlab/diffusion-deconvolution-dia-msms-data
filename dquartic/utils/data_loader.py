from typing import Literal
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Literal


class DIAMSDataset(Dataset):
    def __init__(self, ms2_file, ms1_file, normalize: Literal[None, 'minmax'] = None, split: bool = False, split_ratio: float = 0.5):
        # Use memory mapping to load the data
        self.ms2_data = np.load(ms2_file, mmap_mode='r')
        self.ms1_data = np.load(ms1_file, mmap_mode='r')
        
        # Store file paths for potential later use
        self.ms2_file = ms2_file
        self.ms1_file = ms1_file
        
        self.normalize = normalize
        
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
            
            ms2_sample_split_1 = self.ms2_data[split_1_idx]
            ms1_sample_split_1 = self.ms1_data[split_1_idx]
            ms2_sample_split_2 = self.ms2_data[split_2_idx]
            ms1_sample_split_2 = self.ms1_data[split_2_idx]
            
            if self.normalize == 'minmax':
                self.ms2_min = np.min([ms2_sample_split_1.min(), ms2_sample_split_2.min()])
                self.ms2_max = np.max([ms2_sample_split_1.max(), ms2_sample_split_2.max()])
                self.ms1_min = np.min([ms1_sample_split_1.min(), ms1_sample_split_2.min()])
                self.ms1_max = np.max([ms1_sample_split_1.max(), ms1_sample_split_2.max()])
                
                ms2_sample_split_1 = (ms2_sample_split_1 - self.ms2_min) / (self.ms2_max - self.ms2_min)
                ms1_sample_split_1 = (ms1_sample_split_1 - self.ms1_min) / (self.ms1_max - self.ms1_min)
                ms2_sample_split_2 = (ms2_sample_split_2 - self.ms2_min) / (self.ms2_max - self.ms2_min)
                ms1_sample_split_2 = (ms1_sample_split_2 - self.ms1_min) / (self.ms1_max - self.ms1_min)
            
            return (torch.from_numpy(ms2_sample_split_1.astype(np.float32)),
                    torch.from_numpy(ms1_sample_split_1.astype(np.float32)),
                    torch.from_numpy(ms2_sample_split_2.astype(np.float32)),
                    torch.from_numpy(ms1_sample_split_2.astype(np.float32)))
        else:
            ms2_sample = self.ms2_data[idx]
            ms1_sample = self.ms1_data[idx]
            
            if self.normalize == 'minmax':
                self.ms2_min = ms2_sample.min()
                self.ms2_max = ms2_sample.max()
                self.ms1_min = ms1_sample.min()
                self.ms1_max = ms1_sample.max()
                
                ms2_sample = (ms2_sample - self.ms2_min) / (self.ms2_max - self.ms2_min)
                ms1_sample = (ms1_sample - self.ms1_min) / (self.ms1_max - self.ms1_min)
            
            return (torch.from_numpy(ms2_sample.astype(np.float32)),
                    torch.from_numpy(ms1_sample.astype(np.float32)))

