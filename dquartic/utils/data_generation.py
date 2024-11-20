import os
import numpy as np
import pandas as pd
import polars as pl
from scipy.sparse import csr_matrix
import pyarrow as pa
import pyarrow.parquet as pq
import fastparquet as fpq
from tqdm import tqdm


from dquartic.utils.raw_data_parser import SqMassRawLoader


def find_closest_indices(array, values):
    indices = np.searchsorted(array, values)
    indices = np.clip(indices, 0, len(array) - 1)
    left = np.abs(np.array(array)[indices-1] - np.array(values))
    right = np.abs(np.array(array)[indices] - np.array(values))
    return np.where(left < right, indices-1, indices)

def extract_rt_window(sparse_matrix, unique_rt, rt_window):
    start_idx, end_idx = find_closest_indices(unique_rt, [rt_window[0], rt_window[-1]])
    return sparse_matrix[start_idx:end_idx+1, :].toarray()

def create_sparse_matrix(df, rt_values, mz_values, fixed_mz_size=150):
    # Create mappings
    rt_to_index = {rt: i for i, rt in enumerate(rt_values)}
    index_to_rt = {i: rt for i, rt in enumerate(rt_values)}
    
    # Adjust mz_values if necessary
    if len(mz_values) < fixed_mz_size:
        avg_mz_diff = np.diff(mz_values).mean()
        padding_size = (fixed_mz_size - len(mz_values)) // 2
        padding_values = np.linspace(mz_values[0] - avg_mz_diff * padding_size, 
                                     mz_values[-1] + avg_mz_diff * padding_size, 
                                     padding_size)
        mz_values = np.concatenate([padding_values, mz_values, padding_values])
    
    mz_to_index = {mz: i for i, mz in enumerate(mz_values)}
    index_to_mz = {i: mz for i, mz in enumerate(mz_values)}
    
    # Create lists of RT and m/z values
    rt_values = list(rt_to_index.keys())
    mz_values = list(mz_to_index.keys())

    # Create repeated arrays using NumPy
    rt_repeated = np.repeat(rt_values, len(mz_values))
    mz_repeated = np.tile(mz_values, len(rt_values))

    # Create a DataFrame with all combinations of RT and m/z
    tmp_df = pl.DataFrame({
        'RETENTION_TIME': rt_repeated,
        'mz': mz_repeated,
    })

    # Add RT and m/z indices
    tmp_df = tmp_df.with_columns([
        pl.col('RETENTION_TIME').replace(rt_to_index).alias('rt_index'),
        pl.col('mz').replace(mz_to_index).alias('mz_index')
    ])

    # Sort the DataFrame
    tmp_df = tmp_df.sort(['RETENTION_TIME', 'mz'])

    # join tmp_ms1_filt with df
    df = tmp_df.join(df, on=["RETENTION_TIME", "mz"], how="left").fill_null(0)
    
    # Group by RT and m/z indices, and aggregate intensity
    df = df.group_by(['rt_index', 'mz_index']).agg(pl.sum('intensity'))
    
    # Create sparse matrix
    row_indices = df['rt_index'].to_numpy()
    col_indices = df['mz_index'].to_numpy()
    data = df['intensity'].to_numpy()
    
    sparse_matrix = csr_matrix((data, (row_indices, col_indices)), 
                               shape=(len(rt_values), fixed_mz_size))
    
    return sparse_matrix

def process_ms_data(ms_data, windows, fixed_mz_size):
    unique_rt = ms_data["RETENTION_TIME"].unique().sort()
    unique_mz = ms_data["mz"].unique().sort().drop_nulls()

    sparse_matrix = create_sparse_matrix(ms_data, unique_rt, unique_mz, fixed_mz_size)

    return sparse_matrix, unique_rt, unique_mz

def create_parquet_data(input_file: str, current_iso, slice_ms1, slice_ms2, window, unique_mz, unique_mz_ms2):
    data = []
    slice_data = {
        'file': os.path.basename(input_file),
        'slice_index': i,
        'mz_isolation_target': current_iso['ISOLATION_TARGET'],
        'mz_start': current_iso['mzStart'],
        'mz_end': current_iso['mzEnd'],
        'rt_start': window[0],
        'rt_end': window[-1],
        'ms1_data': slice_ms1.flatten().astype(np.float32),
        'ms2_data': slice_ms2.flatten().astype(np.float32),
        'ms1_shape': list(slice_ms1.shape),
        'ms2_shape': list(slice_ms2.shape),
        'rt_values': np.array(window).astype(np.float32),
        'mz_values_ms1': unique_mz.to_numpy().astype(np.float32),
        'mz_values_ms2': unique_mz_ms2.to_numpy().astype(np.float32),
    }
    data.append(slice_data)
        
    # Define the schema explicitly
    schema = pa.schema([
        ('file', pa.string()),
        ('slice_index', pa.int64()),
        ('mz_isolation_target', pa.float64()),
        ('mz_start', pa.float64()),
        ('mz_end', pa.float64()),
        ('rt_start', pa.float64()),
        ('rt_end', pa.float64()),
        ('ms1_data', pa.list_(pa.float32())),
        ('ms2_data', pa.list_(pa.float32())),
        ('ms1_shape', pa.list_(pa.int64())),
        ('ms2_shape', pa.list_(pa.int64())),
        ('rt_values', pa.list_(pa.float32())),
        ('mz_values_ms1', pa.list_(pa.float32())),
        ('mz_values_ms2', pa.list_(pa.float32()))
    ])
    
    return pa.Table.from_pylist(data, schema=schema)

def write_to_parquet(table, filename):
    
    # Convert list columns to numpy arrays
    for col in ['ms1_data', 'ms2_data', 'rt_values', 'mz_values_ms1', 'mz_values_ms2']:
        if col in table.columns:
            table[col] = table[col].apply(list)
    
    if os.path.exists(filename):
        # If the file exists, append to it
        # with pq.ParquetWriter(filename, table.schema, append=True) as writer:
            # writer.write_table(table)
        fpq.write(filename, table, append=True, object_encoding='json')
    else:
        # If the file doesn't exist, create it
        # pq.write_table(table, filename)
        fpq.write(filename, table, object_encoding='json')


def generate_data_slices(input_file, output_file, window_size=34, sliding_step=5, mz_ppm_tol=10, bin_mz=True, mz_bin_ppm_tol=50, ms1_fixed_mz_size=150, ms2_fixed_mz_size=80_000):

    loader = SqMassRawLoader(input_file)
    loader.load_all_data()

    # Get unique and sorted RETENTION_TIME values
    unique_sorted_rt = pl.concat([loader.ms1_data['RETENTION_TIME'].unique().sort(), loader.ms2_data['RETENTION_TIME'].unique().sort()]).unique().sort()

    # Create sliding windows with overlap
    windows = []
    num_points = len(unique_sorted_rt)

    for start in range(0, num_points, sliding_step):
        end = start + window_size
        
        # Check if there's enough data for a full window
        if end <= num_points:
            window = unique_sorted_rt[start:end].to_list()
            windows.append(window)
            
    schema = pa.schema([
        ('file', pa.string()),
        ('slice_index', pa.int64()),
        ('mz_isolation_target', pa.float64()),
        ('mz_start', pa.float64()),
        ('mz_end', pa.float64()),
        ('rt_start', pa.float64()),
        ('rt_end', pa.float64()),
        ('ms1_data', pa.list_(pa.float32())),
        ('ms2_data', pa.list_(pa.float32())),
        ('ms1_shape', pa.list_(pa.int64())),
        ('ms2_shape', pa.list_(pa.int64())),
        ('rt_values', pa.list_(pa.float32())),
        ('mz_values_ms1', pa.list_(pa.float32())),
        ('mz_values_ms2', pa.list_(pa.float32()))
    ])
            
    pq_writer = pq.ParquetWriter(output_file, schema=schema)

    total_iterations = len(loader.iso_win_info)
    for idx, current_iso in tqdm(loader.iso_win_info.to_pandas().iterrows(), total=total_iterations, desc="Processing isolation windows"):

        ms1_tgt = loader.extract_ms1_slice(current_iso, mz_ppm_tol, bin_mz, mz_bin_ppm_tol)
        ms2_tgt = loader.extract_ms2_slice(current_iso, bin_mz, mz_bin_ppm_tol)

        # Put both MS1 and MS2 RETENTION_TIME values on the same grid
        rt_ms1 = ms1_tgt['RETENTION_TIME']
        rt_ms2 = ms2_tgt['RETENTION_TIME']
        unique_rt = pl.concat([rt_ms1, rt_ms2]).unique().sort().to_frame()

        # Reindex MS1/MS2 DataFrame to align with the common grid
        ms1_tgt = unique_rt.join(ms1_tgt, on="RETENTION_TIME", how="left")
        ms2_tgt = unique_rt.join(ms2_tgt, on="RETENTION_TIME", how="left")

        for window in windows:

            # Process MS1 data
            sparse_matrix, unique_rt, unique_mz = process_ms_data(ms1_tgt, windows, fixed_mz_size=ms1_fixed_mz_size)
            slice_ms1 = extract_rt_window(sparse_matrix, unique_rt, window)

            # Process MS2 data
            sparse_matrix_ms2, _, unique_mz_ms2 = process_ms_data(ms2_tgt, windows, fixed_mz_size=ms2_fixed_mz_size)
            slice_ms2 = extract_rt_window(sparse_matrix_ms2, unique_rt, window)

            # Create Parquet data
            table = create_parquet_data(input_file, current_iso, slice_ms1, slice_ms2, window, unique_mz, unique_mz_ms2)

            # Write to Parquet file
            pq_writer.write(table)
            
            # Clear variables to free up memory
            del sparse_matrix, sparse_matrix_ms2, slice_ms1, slice_ms2, table
        
        del ms1_tgt, ms2_tgt
            
    pq_writer.close()
    
