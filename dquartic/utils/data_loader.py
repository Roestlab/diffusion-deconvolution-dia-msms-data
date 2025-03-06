from typing import Literal
import random
import numpy as np
import duckdb
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Literal


class DIAMSDataset(Dataset):
    """
    A dataset class for loading DIA-MS data.

    Args:
        parquet_directory (str): The directory containing the Parquet files. Mutually exclusive with `ms2_file` and `ms1_file`.
        ms2_file (str): The file path to the MS2 data. Mutually exclusive with `parquet_directory`.
        ms1_file (str): The file path to the MS1 data. Mutually exclusive with `parquet_directory`.
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

    def __init__(self, parquet_directory=None, ms2_file=None, ms1_file=None, normalize: Literal[None, "minmax"] = None):
        if parquet_directory is None and ms1_file is not None and ms2_file is not None:
            self.ms2_data = np.load(ms2_file, mmap_mode="r")
            self.ms1_data = np.load(ms1_file, mmap_mode="r")
            self.data_type = "npy"
            print(f"Info: Loaded  {len(self.ms2_data)} MS2 slice samples and {len(self.ms1_data)} MS1 slice samples from NPY files.")
        elif parquet_directory is not None and ms1_file is None and ms2_file is None:
            self.meta_df = self.read_parquet_meta(parquet_directory)
            self.parquet_directory = parquet_directory  # Store the directory path
            self.data_type = "parquet"
            print(f"Info: Loaded {len(self.meta_df)} MS2 slice samples and MS1 slice samples from Parquet files.")
        else:
            raise ValueError(f"Invalid input data arguments. Please provide either a `parquet_directory` or `ms2_file` and `ms1_file`. Got parquet_directory={parquet_directory}, ms2_file={ms2_file}, ms1_file={ms1_file}.")

        self.normalize = normalize
        self.used_pairs = set()
        self.epoch_reset = False

    def __len__(self):
        if self.data_type=="parquet":
            return len(self.meta_df)
        else:
            return len(self.ms2_data)

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset.
        Args:
            idx (int): The index of the item to retrieve.
        Returns:
            tuple: A tuple containing four torch tensors representing the MS2 and MS1 samples.
        """
        if self.data_type == "npy":
            ms2_sample_split_1, ms1_sample_split_1, ms2_sample_split_2, ms1_sample_split_2 = self._get_npy_pair()
        elif self.data_type == "parquet":
            ms2_sample_split_1, ms1_sample_split_1, ms2_sample_split_2, ms1_sample_split_2 = self._get_parquet_pair()

        if self.normalize == "minmax":
            self.ms2_min = np.min([ms2_sample_split_1.min(), ms2_sample_split_2.min()])
            self.ms2_max = np.max([ms2_sample_split_1.max(), ms2_sample_split_2.max()])
            self.ms1_min = np.min([ms1_sample_split_1.min()])
            self.ms1_max = np.max([ms1_sample_split_1.max()])

            ms2_sample_split_1 = (ms2_sample_split_1 - self.ms2_min) / (self.ms2_max - self.ms2_min)
            ms1_sample_split_1 = (ms1_sample_split_1 - self.ms1_min) / (self.ms1_max - self.ms1_min)
            ms2_sample_split_2 = (ms2_sample_split_2 - self.ms2_min) / (self.ms2_max - self.ms2_min)
            ms1_sample_split_2 = (ms1_sample_split_2 - self.ms1_min) / (self.ms1_max - self.ms1_min)
        else:
            raise ValueError("Invalid normalization method. Valid options are: None, 'minmax'.")

        return (
            torch.from_numpy(ms2_sample_split_1.astype(np.float32)),
            torch.from_numpy(ms1_sample_split_1.astype(np.float32)),
            torch.from_numpy(ms2_sample_split_2.astype(np.float32)),
            torch.from_numpy(ms1_sample_split_2.astype(np.float32)),
        )
        
    def reset_epoch(self):
        """Reset the used pairs at the start of each epoch."""
        self.used_pairs.clear()
        self.epoch_reset = True
        
    def read_parquet_meta(self,parquet_directory):
        """
        Read the metadata from the Parquet files.
        Args:
            parquet_directory (str): The directory containing the Parquet files.
        Returns:
            pandas.DataFrame: A DataFrame containing the metadata.
        """
        query = f"""
            SELECT slice_index, mz_isolation_target, mz_start, mz_end, rt_start, rt_end
            FROM '{parquet_directory}/*.parquet'
        """
        meta_df = duckdb.query(query).to_df()

        return meta_df
    
    def _get_npy_pair(self):
        """Get a random pair of MS2 samples from NPY files."""
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
        
        return ms2_sample_split_1, ms1_sample_split_1, ms2_sample_split_2, ms1_sample_split_2
    
    def _get_parquet_pair(self):
        """Get a random pair of MS2 samples from Parquet files."""
        while True:
            idx_1 = random.randint(0, len(self.meta_df) - 1)
            row_pair_1 = self.meta_df.iloc[idx_1]
            
            idx_2 = random.randint(0, len(self.meta_df) - 1)
            row_pair_2 = self.meta_df.iloc[idx_2]
            
            if row_pair_1['mz_isolation_target'] == row_pair_2['mz_isolation_target'] and row_pair_1['slice_index'] == row_pair_2['slice_index']:
                continue

            ms1_sample_split_1, ms2_sample_split_1 = self._get_parquet_data(row_pair_1)
            
            ms1_sample_split_2, ms2_sample_split_2 = self._get_parquet_data(row_pair_2)

            pair = tuple(sorted((idx_1, idx_2)))

            if pair in self.used_pairs:
                continue

            self.used_pairs.add(pair)
            break

        return ms2_sample_split_1, ms1_sample_split_1, ms2_sample_split_2, ms1_sample_split_2
           
    def _get_parquet_data(self, row):
        """Get the MS2 and MS1 samples from Parquet files."""
        query = f"""
            SELECT 
                ms2_data, 
                ms1_data, 
                ms2_shape, 
                ms1_shape 
            FROM '{self.parquet_directory}/*.parquet'
            WHERE slice_index = {row['slice_index']}
            AND mz_isolation_target = {row['mz_isolation_target']}
            AND mz_start = {row['mz_start']}
            AND mz_end = {row['mz_end']}
            AND rt_start = {row['rt_start']}
            AND rt_end = {row['rt_end']}
            LIMIT 1
        """
        result = duckdb.query(query).fetchall()

        ms2_data, ms1_data, ms2_shape, ms1_shape = result[0]

        ms2_data = np.array(ms2_data).reshape(ms2_shape)
        ms1_data = np.array(ms1_data).reshape(ms1_shape)

        return ms1_data, ms2_data
