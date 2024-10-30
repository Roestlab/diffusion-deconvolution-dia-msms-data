import numpy as np
import pandas as pd
import sqlite3
import polars as pl
import zlib
import struct

# set global plotting backend for pandas
pd.options.plotting.backend = "ms_plotly" # one of: "ms_bokeh" "ms_matplotlib" "ms_plotly"

from plotly.subplots import make_subplots

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
            .with_columns(pl.col("DATA").map_elements(self.decompress_data, return_dtype=pl.List(pl.Float64)).alias("DECOMPRESSED_DATA"))
            .drop(["DATA_TYPE", "DATA"])
        )
        ms_data = ms_data.pivot(on=["DATA_TYPE_STR"], index=["SPECTRUM_ID", "NATIVE_ID", "RETENTION_TIME"], values="DECOMPRESSED_DATA")
        ms_data = ms_data.explode(["mz", "intensity"])
        return ms_data

    def load_all_data(self):
        self.load_isolation_window_info()
        self.load_spectrum_isolation_map()
        self.ms1_data = self.load_ms_data(1)
        self.ms2_data = self.load_ms_data(2)

    def extract_ms_slice(self, rt_tgt, rt_win, mz_iso_win_idx):
        ms1_slice = self.ms1_data.filter(
            (pl.col("mz") >= self.iso_win_info["mzStart"][mz_iso_win_idx])
            & (pl.col("mz") <= self.iso_win_info["mzEnd"][mz_iso_win_idx])
            & (pl.col("RETENTION_TIME") >= rt_tgt - rt_win/2)
            & (pl.col("RETENTION_TIME") <= rt_tgt + rt_win/2)
        ).with_columns(mslevel=1)

        spectrum_ids = (
            self.spec_id_iso_map.filter(pl.col("ISOLATION_TARGET") == self.iso_win_info["ISOLATION_TARGET"][mz_iso_win_idx])
            .select("SPECTRUM_ID")
            .to_numpy()
            .flatten()
        )
        
        print(f"Info: There are {len(spectrum_ids)} MS2 spectra in the {self.iso_win_info['ISOLATION_TARGET'][mz_iso_win_idx]} isolation window")
        
        ms2_slice = self.ms2_data.filter(
            (pl.col("SPECTRUM_ID").is_in(spectrum_ids))
            & (pl.col("RETENTION_TIME") >= rt_tgt - rt_win/2)
            & (pl.col("RETENTION_TIME") <= rt_tgt + rt_win/2)
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
    
    def plot_xic_peakmap(self, ms1_ms2_slice, rt_tgt, rt_win, mz_iso_win_idx):
        p1 = ms1_ms2_slice.to_pandas().plot(
            kind="chromatogram",
            x="RETENTION_TIME",
            y="intensity",
            by="mslevel",
            title="MS1 Chromatogram",
            aggregate_duplicates=False,
            show_plot=False,
        )

        p2 = ms1_ms2_slice.to_pandas().plot(
            kind="peakmap",
            x="RETENTION_TIME",
            y="mz",
            z="normalized_intensity",
            by="mslevel",
            title="MS1 PeakMap",
            z_log_scale=False,
            bin_peaks=True,
            show_plot=False,
        )

        plot_list = [p1, p2]

        fig = make_subplots(rows=1, cols=len(plot_list), subplot_titles=["XIC", "PeakMap"])
        for idx, f in enumerate(plot_list):
            for trace in f.data:
                fig.add_trace(trace, row=1, col=idx + 1)
                fig.update_xaxes(title_text="Retention Time", row=1, col=idx + 1)
                if idx == 0:
                    fig.update_yaxes(title_text="Intensity", row=1, col=idx + 1)
                else:
                    fig.update_yaxes(title_text="mass-to-charge", row=1, col=idx + 1)
                fig.update_layout(
                    title=f"Extraction: mz isolation target = {self.iso_win_info['ISOLATION_TARGET'][mz_iso_win_idx]} | RT = {rt_tgt}±{rt_win/2}",
                    legend_title_text="MS Level",
                    showlegend=True,
                )

        fig.show()
        return fig
        
    def plot_peakmap_3d(self, ms1_ms2_slice, rt_tgt, rt_win, mz_iso_win_idx):
        fig = ms1_ms2_slice.to_pandas().plot(kind="peakmap", x="RETENTION_TIME", y="mz", z="normalized_intensity", by="mslevel", title=f"Extraction: mz isolation target = {self.iso_win_info['ISOLATION_TARGET'][mz_iso_win_idx]} | RT = {rt_tgt}±{rt_win/2}", z_log_scale=False, bin_peaks=False, show_plot=False, plot_3d=True, width=1000, height=800)
        fig.update_layout(legend_title_text="MS Level")

        fig.show()
        return fig