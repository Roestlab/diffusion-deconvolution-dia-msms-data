import duckdb
import time
import os
import psutil  # Used to monitor memory usage
import pandas as pd

def run_query_with_limit(limit):
    try:
        # Construct the DuckDB query with the specified LIMIT
        query = f"""
        SELECT slice_index, mz_isolation_target, mz_start, mz_end, rt_start, rt_end
        FROM './*.parquet'
        LIMIT {limit}
        """

        # Record start time
        start_time = time.time()

        # Record initial memory usage (RSS)
        process = psutil.Process(os.getpid())
        start_memory_rss = process.memory_info().rss  # Resident Set Size in bytes

        # Execute the query
        df = duckdb.query(query).df()

        # Record end time and memory usage (RSS)
        end_time = time.time()
        end_memory_rss = process.memory_info().rss

        # Calculate execution time and memory usage
        execution_time = end_time - start_time
        memory_used_rss = end_memory_rss - start_memory_rss

        # Print results
        print(f"Limit: {limit}")
        print(f"Shape of data: {df.shape}")
        print(f"Execution Time: {execution_time:.4f} seconds")
        print(f"Memory Used (RSS): {memory_used_rss / (1024 * 1024):.2f} MB")  # Convert to MB
        print("-" * 30)

    except Exception as e:
        print(f"Error with limit {limit}: {e}")

if __name__ == "__main__":
    limits = [5, 50, 500, 5000, 50_000, 500_000]
    for limit in limits:
        run_query_with_limit(limit)

