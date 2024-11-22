from memory_profiler import profile
import numpy as np
import pandas as pd
import sqlite3
import polars as pl
import zlib
import struct

class SqMassRawLoader:
    def __init__(self, input_file):
        self.input_file = input_file
        self.conn = sqlite3.connect(input_file)
        self.iso_win_info = None
        self.spec_id_iso_map = None
        self.ms1_data = None
        self.ms2_data = None

    def load_isolation_window_info(self):
        query = """
        SELECT DISTINCT
        ISOLATION_TARGET,
        ISOLATION_LOWER,
        ISOLATION_UPPER
        FROM PRECURSOR
        INNER JOIN SPECTRUM ON SPECTRUM.ID = PRECURSOR.SPECTRUM_ID
        INNER JOIN DATA ON DATA.SPECTRUM_ID = SPECTRUM.ID
        WHERE PRECURSOR.SPECTRUM_ID IS NOT NULL
        ORDER BY ISOLATION_TARGET
        """
        self.iso_win_info = pl.read_database(query, self.conn).with_columns(
            mzStart=pl.col("ISOLATION_TARGET") - pl.col("ISOLATION_LOWER"),
            mzEnd=pl.col("ISOLATION_TARGET") + pl.col("ISOLATION_UPPER"),
        )

    def load_spectrum_isolation_map(self):
        query = """
        SELECT
        PRECURSOR.SPECTRUM_ID,
        ISOLATION_TARGET
        FROM PRECURSOR
        INNER JOIN SPECTRUM ON SPECTRUM.ID = PRECURSOR.SPECTRUM_ID
        WHERE PRECURSOR.SPECTRUM_ID IS NOT NULL
        ORDER BY ISOLATION_TARGET
        """
        self.spec_id_iso_map = pl.read_database(query, self.conn)

    @staticmethod
    def decompress_data(binary_data):
        try:
            tmp = zlib.decompress(binary_data)
            result = struct.unpack(f"<{len(tmp) // 8}d", tmp)
            return list(result)
        except Exception as e:
            print(f"Error decompressing data: {e}")
            return None

    def load_ms_data(self, ms_level):
        query = f"""
        SELECT SPECTRUM_ID, NATIVE_ID, RETENTION_TIME, COMPRESSION, DATA_TYPE, DATA 
        FROM DATA 
        INNER JOIN SPECTRUM ON SPECTRUM.ID = DATA.SPECTRUM_ID 
        WHERE MSLEVEL=={ms_level}
        """
        ms_data = pl.read_database(query, self.conn)
        ms_data = (
            ms_data.with_columns(
                pl.col("DATA_TYPE")
                .map_elements(lambda x: "mz" if x == 0 else "intensity", return_dtype=pl.String)
                .alias("DATA_TYPE_STR")
            )
            .with_columns(
                pl.col("DATA")
                .map_elements(self.decompress_data, return_dtype=pl.List(pl.Float64))
                .alias("DECOMPRESSED_DATA")
            )
            .drop(["DATA_TYPE", "DATA"])
        )
        ms_data = ms_data.pivot(
            on=["DATA_TYPE_STR"],
            index=["SPECTRUM_ID", "NATIVE_ID", "RETENTION_TIME"],
            values="DECOMPRESSED_DATA",
        )
        ms_data = ms_data.explode(["mz", "intensity"])
        return ms_data

    @profile
    def load_all_data(self):
        self.load_isolation_window_info()
        self.load_spectrum_isolation_map()
        self.ms1_data = self.load_ms_data(1)
        self.ms2_data = self.load_ms_data(2)

    @profile
    def extract_ms1_slice(
        self, tgt_mz_frame, ppm_tol: int = 10, bin_mz: bool = True, bin_ppm_tol: int = 50
    ):
        target_mz = (
            self.spec_id_iso_map.filter(
                pl.col("ISOLATION_TARGET") == tgt_mz_frame["ISOLATION_TARGET"]
            )
            .select("ISOLATION_TARGET")
            .unique()
            .to_numpy()
            .flatten()
        )
        tolerance = target_mz * ppm_tol / 1_000_000  # 10 ppm tolerance
        lower_bound = target_mz - tolerance
        upper_bound = target_mz + tolerance

        lower_bound, upper_bound = tgt_mz_frame[["mzStart", "mzEnd"]].unique().flatten()

        ms1_tgt = self.ms1_data.filter(
            (pl.col("mz") >= lower_bound) & (pl.col("mz") <= upper_bound)
        ).with_columns(mslevel=1)

        if bin_mz:
            ms1_tgt = ms1_tgt.group_by("mslevel").map_groups(self.bin_ppm)

            # Compute the average mz for each bin
            average_mz = ms1_tgt.group_by(["mslevel", "mz_bin"]).agg(
                pl.col("mz").mean().alias("average_mz")
            )

            # Join the average mz back to the original DataFrame
            ms1_tgt = ms1_tgt.join(average_mz, on=["mslevel", "mz_bin"])

            # Rename mz as mz_org and average_mz as mz
            ms1_tgt = ms1_tgt.rename({"mz": "mz_org", "average_mz": "mz"})

        return ms1_tgt

    @profile
    def extract_ms2_slice(self, tgt_mz_frame, bin_mz: bool = True, bin_ppm_tol: int = 50):
        spectrum_ids = (
            self.spec_id_iso_map.filter(
                pl.col("ISOLATION_TARGET") == tgt_mz_frame["ISOLATION_TARGET"]
            )
            .select("SPECTRUM_ID")
            .to_numpy()
            .flatten()
        )
        ms2_tgt = self.ms2_data.filter((pl.col("SPECTRUM_ID").is_in(spectrum_ids))).with_columns(
            mslevel=2
        )

        if bin_mz:
            ms2_tgt = ms2_tgt.group_by("mslevel").map_groups(self.bin_ppm)

            # Compute the average mz for each bin
            average_mz = ms2_tgt.group_by(["mslevel", "mz_bin"]).agg(
                pl.col("mz").mean().alias("average_mz")
            )

            # Join the average mz back to the original DataFrame
            ms2_tgt = ms2_tgt.join(average_mz, on=["mslevel", "mz_bin"])

            # Rename mz as mz_org and average_mz as mz
            ms2_tgt = ms2_tgt.rename({"mz": "mz_org", "average_mz": "mz"})

        return ms2_tgt

    def extract_ms_slice(self, rt_tgt, rt_win, mz_iso_win_idx):
        ms1_slice = self.ms1_data.filter(
            (pl.col("mz") >= self.iso_win_info["mzStart"][mz_iso_win_idx])
            & (pl.col("mz") <= self.iso_win_info["mzEnd"][mz_iso_win_idx])
            & (pl.col("RETENTION_TIME") >= rt_tgt - rt_win / 2)
            & (pl.col("RETENTION_TIME") <= rt_tgt + rt_win / 2)
        ).with_columns(mslevel=1)

        spectrum_ids = (
            self.spec_id_iso_map.filter(
                pl.col("ISOLATION_TARGET") == self.iso_win_info["ISOLATION_TARGET"][mz_iso_win_idx]
            )
            .select("SPECTRUM_ID")
            .to_numpy()
            .flatten()
        )

        print(
            f"Info: There are {len(spectrum_ids)} MS2 spectra in the {self.iso_win_info['ISOLATION_TARGET'][mz_iso_win_idx]} isolation window"
        )

        ms2_slice = self.ms2_data.filter(
            (pl.col("SPECTRUM_ID").is_in(spectrum_ids))
            & (pl.col("RETENTION_TIME") >= rt_tgt - rt_win / 2)
            & (pl.col("RETENTION_TIME") <= rt_tgt + rt_win / 2)
        ).with_columns(mslevel=2)

        ms1_ms2_slice = pl.concat([ms1_slice, ms2_slice])
        ms1_ms2_slice = ms1_ms2_slice.group_by("mslevel").map_groups(
            lambda group: group.with_columns(
                pl.col("intensity")
                .map_batches(lambda x: (x - x.min()) / (x.max() - x.min()))
                .alias("normalized_intensity")
            )
        )

        return ms1_ms2_slice

    @staticmethod
    def bin_ppm(df: pl.DataFrame, ppm: int = 50) -> pl.DataFrame:
        # Filter for non-null mz values
        mz_values = df["mz"]
        reference_mz = mz_values.min()  # Smallest mz as the reference
        bin_edges = reference_mz * (1 + np.arange(0, len(mz_values) + 1) * ppm / 1e6)
        # Assign bins
        bins = pd.cut(mz_values.to_numpy(), bins=bin_edges, labels=False)
        # Add the bins as a new column
        return df.with_columns(pl.Series(name="mz_bin", values=bins))
