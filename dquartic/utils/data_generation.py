import os
import datetime
import numpy as np
import pandas as pd
import polars as pl
from scipy.sparse import csr_matrix
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from memory_profiler import profile
import gc
import tracemalloc
import psutil
import concurrent.futures


from dquartic.utils.raw_data_parser import SqMassRawLoader


def log_memory_usage(batch_i):
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return f"Batch {batch_i} memory usage: {memory_info.rss / 1024 / 1024:.2f} MB"


def find_closest_indices(array, values):
    indices = np.searchsorted(array, values)
    indices = np.clip(indices, 0, len(array) - 1)
    left = np.abs(np.array(array)[indices - 1] - np.array(values))
    right = np.abs(np.array(array)[indices] - np.array(values))
    return np.where(left < right, indices - 1, indices)


def extract_rt_window(sparse_matrix, unique_rt, rt_window):
    start_idx, end_idx = find_closest_indices(unique_rt, [rt_window[0], rt_window[-1]])
    return sparse_matrix[start_idx : end_idx + 1, :].toarray()


def create_sparse_matrix(df, rt_values, mz_values):
    # Create mappings
    rt_to_index = {rt: i for i, rt in enumerate(rt_values)}
    index_to_rt = {i: rt for i, rt in enumerate(rt_values)}

    mz_to_index = {mz: i for i, mz in enumerate(mz_values)}
    index_to_mz = {i: mz for i, mz in enumerate(mz_values)}

    # Create lists of RT and m/z values
    rt_values = list(rt_to_index.keys())
    mz_values = list(mz_to_index.keys())

    # Create repeated arrays using NumPy
    rt_repeated = np.repeat(rt_values, len(mz_values))
    mz_repeated = np.tile(mz_values, len(rt_values))

    # Create a DataFrame with all combinations of RT and m/z
    tmp_df = pl.DataFrame(
        {
            "RETENTION_TIME": rt_repeated,
            "mz": mz_repeated,
        }
    )

    # Add RT and m/z indices
    tmp_df = tmp_df.with_columns(
        [
            pl.col("RETENTION_TIME").replace(rt_to_index).alias("rt_index"),
            pl.col("mz").replace(mz_to_index).alias("mz_index"),
        ]
    )

    # Sort the DataFrame
    tmp_df = tmp_df.sort(["RETENTION_TIME", "mz"])

    # join tmp_ms1_filt with df
    df = tmp_df.join(df, on=["RETENTION_TIME", "mz"], how="left").fill_null(0)

    # Group by RT and m/z indices, and aggregate intensity
    df = df.group_by(["rt_index", "mz_index"]).agg(pl.sum("intensity"))

    # Create sparse matrix
    row_indices = df["rt_index"].to_numpy()
    col_indices = df["mz_index"].to_numpy()
    data = df["intensity"].to_numpy()
    
    sparse_matrix = csr_matrix(
        (data, (row_indices, col_indices)), shape=(len(rt_values), len(mz_values))
    )

    return sparse_matrix

def concat_chunked_mzs(list_mz_series):
    """
    Concatenate a list of chunked mz pl.Series values and return indices of second and so on duplicate mz values
    """
    # Concatenate the DataFrames
    unique_mz_ms2 = pl.concat(list_mz_series).to_frame()

    # Convert the "mz" column to a numpy array
    mz_values = unique_mz_ms2["mz"].to_numpy()

    # Find the first occurrence indices of each unique value
    _, first_occurrence_indices = np.unique(mz_values, return_index=True)

    # Initialize a boolean mask for duplicates
    duplicates_mask = np.zeros(len(mz_values), dtype=bool)

    # Mark duplicates by setting the mask to True for all indices except the first occurrence
    for idx in first_occurrence_indices:
        duplicates_mask[idx] = False  # Set first occurrence to False
    duplicates_mask[~np.isin(np.arange(len(mz_values)), first_occurrence_indices)] = True  # Mark all non-first occurrences as duplicates

    # Get the indices of the duplicate rows (True in the mask)
    duplicate_indices = np.where(duplicates_mask)[0]

    return unique_mz_ms2.unique().to_series(), duplicate_indices

# @profile
def process_ms_data(ms_data, windows, is_chunk=False):
    unique_rt = ms_data["RETENTION_TIME"].unique().sort()
    unique_mz = ms_data["mz"].unique().sort().drop_nulls()
    
    sparse_matrix = create_sparse_matrix(ms_data, unique_rt, unique_mz)
    slices = []
    for window in windows:
        slice_data = extract_rt_window(sparse_matrix, unique_rt.to_numpy(), window)
        if not is_chunk and slice_data.max()==0:
            # If the max value is 0, then the slice is empty. Probably no signal in this window
            slices.append(np.array([]))
        else:
            slices.append(slice_data)

    return slices, unique_rt, unique_mz

def process_ms_data_in_chunks(ms_data, windows, num_chunks=3, threads=3):
    # Define a wrapper function for process_ms_data
    def process_chunk(chunk):
        return process_ms_data(chunk, windows, is_chunk=True)

    sorted_df = ms_data.sort("mz")

    # Split into N chunks
    chunk_size = len(sorted_df) // num_chunks
    chunks = [sorted_df[i * chunk_size : (i + 1) * chunk_size] for i in range(num_chunks)]

    # Handle any remainder (last chunk may contain extra rows)
    if len(sorted_df) % num_chunks != 0:
        chunks[-1] = sorted_df[(num_chunks - 1) * chunk_size :]
        
    with concurrent.futures.ThreadPoolExecutor(threads) as executor:
        results = list(executor.map(process_chunk, chunks))

    # Unpack each slices and group window chunks
    slices_list = [result[0] for result in results]
    slices_ms2_chunks = []
    for slices in zip(*slices_list):
        slices_ms2_chunks.append(list(slices))
        
    unique_mz_ms2_chunks = [res[2] for res in results]

    unique_mz_ms2, duplicate_indices = concat_chunked_mzs(unique_mz_ms2_chunks)

    slices_ms2 = []
    for ms2_slice_chunk in slices_ms2_chunks:
        # print(f"Concatenating {len(ms2_slice_chunk)} slices", flush=True)
        # print(f"Shape of each slice: {[slice.shape for slice in ms2_slice_chunk]}", flush=True)
        
        tmp = np.concatenate((ms2_slice_chunk), axis=1)
        if duplicate_indices.size > 0:
            tmp = np.delete(tmp, duplicate_indices, axis=1)
        if tmp.max()==0:
            # If the max value is 0, then the slice is empty. Probably no signal in this window
            slices_ms2.append(np.array([]))
        else:
            slices_ms2.append(tmp)
    
    return slices_ms2, unique_mz_ms2


def create_parquet_data(
    input_file: str, current_iso, slices_ms1, slices_ms2, windows, unique_mz, unique_mz_ms2
):
    data = []
    for i, (slice_ms1, slice_ms2, window) in enumerate(zip(slices_ms1, slices_ms2, windows)):
        if slice_ms1.size == 0 or slice_ms2.size == 0:
            continue
        
        slice_data = {
            "file": os.path.basename(input_file),
            "slice_index": i,
            "mz_isolation_target": current_iso["ISOLATION_TARGET"],
            "mz_start": current_iso["mzStart"],
            "mz_end": current_iso["mzEnd"],
            "rt_start": window[0],
            "rt_end": window[-1],
            "ms1_data": slice_ms1.flatten().astype(np.float32),
            "ms2_data": slice_ms2.flatten().astype(np.float32),
            "ms1_shape": list(slice_ms1.shape),
            "ms2_shape": list(slice_ms2.shape),
            "rt_values": np.array(window).astype(np.float32),
            "mz_values_ms1": unique_mz.to_numpy().astype(np.float32),
            "mz_values_ms2": unique_mz_ms2.to_numpy().astype(np.float32),
        }
        data.append(slice_data)

    # Define the schema explicitly
    schema = pa.schema(
        [
            ("file", pa.string()),
            ("slice_index", pa.int64()),
            ("mz_isolation_target", pa.float64()),
            ("mz_start", pa.float64()),
            ("mz_end", pa.float64()),
            ("rt_start", pa.float64()),
            ("rt_end", pa.float64()),
            ("ms1_data", pa.list_(pa.float32())),
            ("ms2_data", pa.list_(pa.float32())),
            ("ms1_shape", pa.list_(pa.int64())),
            ("ms2_shape", pa.list_(pa.int64())),
            ("rt_values", pa.list_(pa.float32())),
            ("mz_values_ms1", pa.list_(pa.float32())),
            ("mz_values_ms2", pa.list_(pa.float32())),
        ]
    )

    return pa.Table.from_pylist(data, schema=schema)


@profile
def generate_data_slices(
    input_file,
    output_file,
    isolation_window_index,
    window_size=34,
    sliding_step=5,
    mz_ppm_tol=10,
    bin_mz=True,
    ms1_fixed_mz_size=150,
    ms2_fixed_mz_size=30_000,
    batch_size=500,
    num_chunks=3, 
    threads=3
):

    loader = SqMassRawLoader(input_file)
    loader.load_all_data()

    # Get unique and sorted RETENTION_TIME values
    unique_sorted_rt = (
        pl.concat(
            [
                loader.ms1_data["RETENTION_TIME"].unique().sort(),
                loader.ms2_data["RETENTION_TIME"].unique().sort(),
            ]
        )
        .unique()
        .sort()
    )

    # Create sliding windows with overlap
    windows = []
    num_points = len(unique_sorted_rt)

    for start in range(0, num_points, sliding_step):
        end = start + window_size

        # Check if there's enough data for a full window
        if end <= num_points:
            window = unique_sorted_rt[start:end].to_list()
            windows.append(window)
    print(f"[{datetime.datetime.now().isoformat()}] Number of RT window slcies: {len(windows)}")

    schema = pa.schema(
        [
            ("file", pa.string()),
            ("slice_index", pa.int64()),
            ("mz_isolation_target", pa.float64()),
            ("mz_start", pa.float64()),
            ("mz_end", pa.float64()),
            ("rt_start", pa.float64()),
            ("rt_end", pa.float64()),
            ("ms1_data", pa.list_(pa.float32())),
            ("ms2_data", pa.list_(pa.float32())),
            ("ms1_shape", pa.list_(pa.int64())),
            ("ms2_shape", pa.list_(pa.int64())),
            ("rt_values", pa.list_(pa.float32())),
            ("mz_values_ms1", pa.list_(pa.float32())),
            ("mz_values_ms2", pa.list_(pa.float32())),
        ]
    )

    pq_writer = pq.ParquetWriter(output_file, schema=schema)
    
    current_iso = loader.iso_win_info.to_pandas().iloc[isolation_window_index]

    print(
        f"[{datetime.datetime.now().isoformat()}] {isolation_window_index} of {len(loader.iso_win_info)} Processing isolation target {current_iso['ISOLATION_TARGET']}"
    )
    ms1_tgt = loader.extract_ms1_slice(current_iso, mz_ppm_tol, bin_mz, ms1_fixed_mz_size)
    ms2_tgt = loader.extract_ms2_slice(current_iso, bin_mz, ms2_fixed_mz_size)

    # Put both MS1 and MS2 RETENTION_TIME values on the same grid
    ms1_tgt = unique_sorted_rt.to_frame().join(ms1_tgt, on="RETENTION_TIME", how="left")
    ms2_tgt = unique_sorted_rt.to_frame().join(ms2_tgt, on="RETENTION_TIME", how="left")

    # List to store tables
    all_tables = []
    batch_counter = 0

    for batch_i in tqdm(range(0, len(windows), batch_size)):
        window_batch = windows[batch_i : batch_i + batch_size]
        print(f"[{datetime.datetime.now().isoformat()}] Processing batch {batch_i} to {batch_i + batch_size}", flush=True)

        # Process MS1 data
        slices_ms1, unique_rt, unique_mz = process_ms_data(ms1_tgt, window_batch)

        # Process MS2 data
        slices_ms2, unique_mz_ms2 = process_ms_data_in_chunks(
            ms2_tgt, window_batch, num_chunks, threads
        )

        # Create Parquet data
        table = create_parquet_data(
            input_file,
            current_iso,
            slices_ms1,
            slices_ms2,
            window_batch,
            unique_mz,
            unique_mz_ms2,
        )

        # Accumulate table
        all_tables.append(table)

        # Clear variables to free up memory
        del slices_ms1, slices_ms2, table

        # Check if all_tables has grown too large
        # Do this to avoid OOM errors
        if len(all_tables) >= 20:
            print(f"[{datetime.datetime.now().isoformat()}] Writing out batch of data...", flush=True)
            # Measure memory usage before deletion
            tracemalloc.start()
            snapshot1 = tracemalloc.take_snapshot()
            intermediate_table = pa.concat_tables(all_tables)
            if batch_counter == 0:
                pq.write_table(intermediate_table, output_file, compression='snappy')
            else:
                with pq.ParquetWriter(output_file, intermediate_table.schema, compression='snappy') as writer:
                    writer.write_table(intermediate_table)
            # Remove the intermediate table from memory
            del intermediate_table
            
            # Clear accumulated tables
            all_tables.clear()
            
            # Force garbage collection to ensure memory release
            gc.collect()
            
            # Measure memory usage after deletion
            snapshot2 = tracemalloc.take_snapshot()
            stats = snapshot2.compare_to(snapshot1, 'lineno')
            print(f"[{datetime.datetime.now().isoformat()}] Memory released: {stats[0].size_diff / 10**9:.2f} GB", flush=True)
            
            batch_counter += 1

    # Measure memory usage before deletion
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()
    # Write remaining tables to Parquet if any
    if all_tables:
        print(f"[{datetime.datetime.now().isoformat()}] Writing out remaining data...", flush=True)
        final_table = pa.concat_tables(all_tables)
        if batch_counter == 0:
            pq.write_table(final_table, output_file, compression='snappy')
        else:
            with pq.ParquetWriter(output_file, final_table.schema, compression='snappy') as writer:
                writer.write_table(final_table)
        del final_table

    # Clear memory
    del all_tables

    del ms1_tgt, ms2_tgt
    
    # Force garbage collection to ensure memory release
    gc.collect()
    
    # Measure memory usage after deletion
    snapshot2 = tracemalloc.take_snapshot()
    stats = snapshot2.compare_to(snapshot1, 'lineno')
    print(f"[{datetime.datetime.now().isoformat()}] Memory released: {stats[0].size_diff / 10**9:.2f} GB", flush=True)

    pq_writer.close()
