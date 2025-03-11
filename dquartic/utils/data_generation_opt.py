import os
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import concurrent.futures
from dquartic.utils.raw_data_parser import SqMassRawLoader


def log_memory_usage(stage):
    """Log memory usage at various stages for debugging."""
    import psutil

    process = os.getpid()
    memory_info = psutil.Process(process).memory_info()
    print(f"[Memory] {stage}: {memory_info.rss / (1024 ** 3):.2f} GB")


def find_closest_indices(array, values):
    """Find the closest indices in a sorted array for given values."""
    indices = np.searchsorted(array, values)
    indices = np.clip(indices, 0, len(array) - 1)
    return indices


def extract_rt_window(sparse_matrix, unique_rt, rt_window):
    """Extract a retention time window from a sparse matrix."""
    start_idx, end_idx = find_closest_indices(unique_rt, [rt_window[0], rt_window[-1]])
    return sparse_matrix[start_idx : end_idx + 1]  # Avoid dense conversion


def create_sparse_matrix(df, rt_values, mz_values):
    """Create a sparse matrix from retention time and m/z values."""
    rt_to_index = {rt: i for i, rt in enumerate(rt_values)}
    mz_to_index = {mz: i for i, mz in enumerate(mz_values)}

    rt_indices = df["RETENTION_TIME"].map(rt_to_index).to_numpy()
    mz_indices = df["mz"].map(mz_to_index).to_numpy()
    intensities = df["intensity"].to_numpy()

    return csr_matrix(
        (intensities, (rt_indices, mz_indices)), shape=(len(rt_values), len(mz_values))
    )


def process_ms_data(ms_data, windows):
    """Process MS data and extract slices for each window."""
    unique_rt = ms_data["RETENTION_TIME"].unique().sort().to_numpy()
    unique_mz = ms_data["mz"].unique().sort().to_numpy()

    sparse_matrix = create_sparse_matrix(ms_data, unique_rt, unique_mz)
    slices = [extract_rt_window(sparse_matrix, unique_rt, window) for window in windows]
    return slices, unique_rt, unique_mz


def process_ms_data_in_chunks(ms_data, windows, num_chunks, threads):
    """Process MS data in chunks to handle large datasets."""
    chunk_size = len(ms_data) // num_chunks
    chunks = [ms_data[i * chunk_size : (i + 1) * chunk_size] for i in range(num_chunks)]

    with concurrent.futures.ProcessPoolExecutor(threads) as executor:
        results = list(executor.map(lambda chunk: process_ms_data(chunk, windows), chunks))

    slices = [result[0] for result in results]
    slices_combined = [np.concatenate(slice_group, axis=1) for slice_group in zip(*slices)]

    unique_mz_combined = np.unique(np.concatenate([result[2] for result in results]))
    return slices_combined, unique_mz_combined


def write_parquet(output_file, schema, tables):
    """Write data to a Parquet file."""
    with pq.ParquetWriter(output_file, schema=schema) as writer:
        for table in tables:
            writer.write_table(table)


def generate_data_slices(
    input_file,
    output_file,
    isolation_window_index,
    window_size=34,
    sliding_step=5,
    mz_ppm_tol=10,
    bin_mz=True,
    ms1_fixed_mz_size=150,
    ms2_fixed_mz_size=30000,
    batch_size=500,
    batch_writing_size=20,
    num_chunks=3,
    threads=3,
):
    loader = SqMassRawLoader(input_file)
    loader.load_all_data()

    unique_sorted_rt = (
        pl.concat([loader.ms1_data["RETENTION_TIME"], loader.ms2_data["RETENTION_TIME"]])
        .unique()
        .sort()
    ).to_numpy()

    windows = [
        unique_sorted_rt[i : i + window_size]
        for i in range(0, len(unique_sorted_rt) - window_size + 1, sliding_step)
    ]

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

    current_iso = loader.iso_win_info.to_pandas().iloc[isolation_window_index]

    ms1_tgt = loader.extract_ms1_slice(current_iso, mz_ppm_tol, bin_mz, ms1_fixed_mz_size)
    ms2_tgt = loader.extract_ms2_slice(current_iso, bin_mz, ms2_fixed_mz_size)

    ms1_tgt = unique_sorted_rt.to_frame().join(ms1_tgt, on="RETENTION_TIME", how="left")
    ms2_tgt = unique_sorted_rt.to_frame().join(ms2_tgt, on="RETENTION_TIME", how="left")

    batch_tables = []
    for batch_i in tqdm(range(0, len(windows), batch_size)):
        batch_windows = windows[batch_i : batch_i + batch_size]

        slices_ms1, _, unique_mz = process_ms_data(ms1_tgt, batch_windows)
        slices_ms2, unique_mz_ms2 = process_ms_data_in_chunks(
            ms2_tgt, batch_windows, num_chunks, threads
        )

        for i, (slice_ms1, slice_ms2, window) in enumerate(
            zip(slices_ms1, slices_ms2, batch_windows)
        ):
            table = pa.Table.from_pydict(
                {
                    "file": os.path.basename(input_file),
                    "slice_index": i,
                    "mz_isolation_target": current_iso["ISOLATION_TARGET"],
                    "mz_start": current_iso["mzStart"],
                    "mz_end": current_iso["mzEnd"],
                    "rt_start": window[0],
                    "rt_end": window[-1],
                    "ms1_data": slice_ms1.flatten().astype(np.float32).tolist(),
                    "ms2_data": slice_ms2.flatten().astype(np.float32).tolist(),
                    "ms1_shape": list(slice_ms1.shape),
                    "ms2_shape": list(slice_ms2.shape),
                    "rt_values": window.astype(np.float32).tolist(),
                    "mz_values_ms1": unique_mz.astype(np.float32).tolist(),
                    "mz_values_ms2": unique_mz_ms2.astype(np.float32).tolist(),
                },
                schema=schema,
            )

            batch_tables.append(table)

        if len(batch_tables) >= batch_writing_size:
            write_parquet(output_file, schema, batch_tables)
            batch_tables.clear()

    if batch_tables:
        write_parquet(output_file, schema, batch_tables)
