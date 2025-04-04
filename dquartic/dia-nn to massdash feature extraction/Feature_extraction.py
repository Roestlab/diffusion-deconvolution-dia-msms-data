
"""
Feature_extraction.py

DIA-NN to MassDASH Feature Extractor

This script processes DIA-NN output files (in .parquet format) and extracts MS1 and MS2 features 
in batch mode for downstream analysis in the diffusion deconvolution-dia-msms-data (D4) pipeline.

Features:
    - Converts DIA-NN parquet files to MassDASH-compatible TSV format.
    - Extracts MS1 and MS2 features in GPU-accelerated batch processing using cuDF.
    - Exports per-batch CSV files and combines them into final output files.
    - Visualizes identification metrics using matplotlib.
    - Configurable via a JSON file (default: dquartic_train_config.json).

Usage:
    python Feature_extraction.py --config dquartic_train_config.json
"""

import os
import json
import time
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cudf  # Ensure you have cuDF installed as part of RAPIDS
from tqdm import tqdm

# These imports should be adjusted based on your project structure
from massdash.loaders import MzMLDataLoader
from massdash.structs import TargetedDIAConfig

# Configure logging for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DIAtoMassProcessor:
    """
    Class to process DIA-NN data and extract MS1/MS2 features for MassDASH.

    The processor performs the following steps:
      1. Converts DIA-NN .parquet files to TSV format for compatibility.
      2. Initializes a MassDASH loader for raw mzML files.
      3. Configures extraction parameters.
      4. Processes peptides in batches with GPU-accelerated DataFrame filtering.
      5. Exports individual batch results and combines them into final output CSV files.
    """
    def __init__(self, report_path, gen_lib_path, mzml_path, batch_size=2000, threads=4, 
                 output_dir="output_features", rt_window=50, mz_tol=20):
        """
        Initialize the DIAtoMassProcessor.

        Args:
            report_path (str): Path to the DIA-NN report file (parquet format).
            gen_lib_path (str): Path to the DIA-NN spectral library file (parquet format).
            mzml_path (str): Path or glob pattern for mzML raw data files.
            batch_size (int, optional): Number of peptides per batch. Defaults to 2000.
            threads (int, optional): Number of threads to use (if applicable). Defaults to 4.
            output_dir (str, optional): Directory to store output features. Defaults to "output_features".
            rt_window (int, optional): Retention time window (seconds) for feature extraction. Defaults to 50.
            mz_tol (int, optional): m/z tolerance (ppm) for matching fragments. Defaults to 20.
        """
        self.report_path = report_path
        self.gen_lib_path = gen_lib_path
        self.mzml_path = mzml_path
        self.batch_size = batch_size
        self.threads = threads
        self.output_dir = output_dir
        self.rt_window = rt_window
        self.mz_tol = mz_tol
        
        # Create output directories for MS1 and MS2 features
        self.ms1_dir = os.path.join(output_dir, "ms1_features")
        self.ms2_dir = os.path.join(output_dir, "ms2_features")
        os.makedirs(self.ms1_dir, exist_ok=True)
        os.makedirs(self.ms2_dir, exist_ok=True)
        
        logging.info(f"Initialized processor with batch size {batch_size} and {threads} threads")
        logging.info(f"Output directories created: {self.ms1_dir}, {self.ms2_dir}")

    def convert_parquet_to_tsv(self):
        """
        Convert DIA-NN parquet files to TSV files compatible with MassDASH.

        This method reads the report and spectral library parquet files, converts them to TSV,
        and renames columns in the spectral library file to match MassDASH expected names.
        Also generates a plot of identifications per file.
        
        Returns:
            tuple: (report_tsv_path, gen_lib_modified_tsv_path)
        """
        logging.info("Converting report.parquet to report.tsv...")
        report_df = pd.read_parquet(self.report_path)
        report_tsv_path = os.path.join(os.path.dirname(self.output_dir), 'report.tsv')
        report_df.to_csv(report_tsv_path, sep='\t', index=False)
        
        logging.info("Converting gen_spec_lib.parquet to gen_lib_modified.tsv...")
        gen_lib_df = pd.read_parquet(self.gen_lib_path)
        
        # Mapping of DIA-NN column names to MassDASH column names.
        column_mapping = {
            'Precursor.Charge': 'PrecursorCharge',
            'Protein.Ids': 'ProteinId',
            'Stripped.Sequence': 'PeptideSequence',
            'Precursor.Mz': 'PrecursorMz',
            'Modified.Sequence': 'ModifiedPeptideSequence',
            'Relative.Intensity': 'LibraryIntensity',
            'Product.Mz': 'ProductMz',
            'Fragment.Type': 'FragmentType',
            'Fragment.Series.Number': 'FragmentSeriesNumber',
            'Fragment.Charge': 'ProductCharge'
        }
        rename_cols = {k: v for k, v in column_mapping.items() if k in gen_lib_df.columns}
        gen_lib_df = gen_lib_df.rename(columns=rename_cols)
        
        gen_lib_modified_tsv_path = os.path.join(os.path.dirname(self.output_dir), 'gen_lib_modified.tsv')
        gen_lib_df.to_csv(gen_lib_modified_tsv_path, sep='\t', index=False)
        
        # Generate identification plot from the report
        self.plot_identifications(report_df)
        
        logging.info(f"Conversion complete. Files saved to {report_tsv_path} and {gen_lib_modified_tsv_path}")
        return report_tsv_path, gen_lib_modified_tsv_path

    def plot_identifications(self, report_df):
        """
        Plot identifications per file (filtered to 1% FDR).

        Args:
            report_df (pandas.DataFrame): DataFrame containing the DIA-NN report.
        
        Returns:
            str: Path to the saved plot image.
        """
        logging.info("Plotting identifications per file at 1% FDR...")
        filtered_df = report_df[report_df["Q.Value"] < 0.01]
        id_counts = filtered_df.groupby("Run").size().reset_index(name="Identifications")
        
        plt.figure(figsize=(10, 5))
        plt.bar(id_counts["Run"], id_counts["Identifications"], color="skyblue")
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("File")
        plt.ylabel("Number of Identifications")
        plt.title("Identifications per File (1% Precursor FDR)")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        
        plot_path = os.path.join(self.output_dir, "identifications_per_file.png")
        plt.savefig(plot_path)
        logging.info(f"Plot saved to {plot_path}")
        return plot_path

    def initialize_loader(self):
        """
        Initialize the MassDASH data loader to access mzML files.

        The loader requires the TSV files produced by convert_parquet_to_tsv.
        """
        logging.info("Initializing MassDASH loader...")
        report_tsv_path = os.path.join(os.path.dirname(self.output_dir), 'report.tsv')
        gen_lib_modified_tsv_path = os.path.join(os.path.dirname(self.output_dir), 'gen_lib_modified.tsv')
        
        self.loader = MzMLDataLoader(
            dataFiles=self.mzml_path,
            rsltsFile=report_tsv_path,
            libraryFile=gen_lib_modified_tsv_path
        )
        logging.info(f"MassDASH loader initialized with mzML path: {self.mzml_path}")

    def configure_extraction(self):
        """
        Configure the extraction parameters for feature extraction.

        Sets the retention time window and m/z tolerance.
        """
        logging.info("Configuring extraction parameters...")
        self.extraction_config = TargetedDIAConfig()
        self.extraction_config.rt_window = self.rt_window
        self.extraction_config.mz_tol = self.mz_tol
        logging.info(f"Extraction configured with RT window: {self.rt_window}s, m/z tolerance: {self.mz_tol} ppm")

    def run_batch_processing(self):
        """
        Process peptides in batches to extract MS1 and MS2 features using GPU acceleration.

        Steps:
            1. Load unique peptide-charge pairs from the report.
            2. Split the peptides into batches based on the configured batch size.
            3. For each peptide, extract features via the MassDASH loader.
            4. Convert the feature data to a GPU DataFrame (cuDF) for accelerated filtering.
            5. Extract and rename MS1 (precursor) and MS2 (fragment) features.
            6. Save each batch as CSV files and combine all batches into final output files.

        Returns:
            dict: Summary of processing including total peptides and output file paths.
        """
        logging.info(f"Loading peptides from {self.report_path}")
        report_df = pd.read_parquet(self.report_path)
        unique_peptides = report_df[['Modified.Sequence', 'Precursor.Charge']].drop_duplicates()
        logging.info(f"Total unique peptides: {len(unique_peptides)}")
        
        # Split unique peptides into batches.
        peptide_batches = np.array_split(unique_peptides, int(np.ceil(len(unique_peptides) / self.batch_size)))
        logging.info(f"Split peptides into {len(peptide_batches)} batches with batch size {self.batch_size}")
        
        ms1_all_batches = []
        ms2_all_batches = []
        
        # Process each batch sequentially
        for batch_idx, batch in enumerate(peptide_batches):
            logging.info(f"\nProcessing batch {batch_idx+1}/{len(peptide_batches)} with {len(batch)} peptides")
            batch_start_time = time.time()
            ms1_list = []
            ms2_list = []
            
            for idx, row in batch.iterrows():
                peptide = row['Modified.Sequence']
                charge = int(row['Precursor.Charge'])
                logging.info(f"  Processing peptide {peptide} (charge {charge}, index in batch: {idx})...")
                
                try:
                    start_extraction = time.time()
                    feature_map_collection = self.loader.loadFeatureMaps(peptide, charge, self.extraction_config)
                    elapsed_extraction = time.time() - start_extraction
                    logging.info(f"    Extraction done in {elapsed_extraction:.2f} seconds.")
                except Exception as e:
                    logging.error(f"    Error loading {peptide} (charge {charge}): {e}")
                    continue
                
                # Process each run's feature map
                for run_name, feature_map in feature_map_collection.items():
                    # Copy feature DataFrame and add identifying columns
                    df = feature_map.feature_df.copy()
                    df['Peptide'] = peptide
                    df['Charge'] = charge
                    df['Run'] = run_name
                    
                    # Convert to cuDF DataFrame for GPU processing
                    gpu_df = cudf.DataFrame.from_pandas(df)
                    
                    # ----- Extract MS1 (Precursor) Data -----
                    ms1_gpu = gpu_df[(gpu_df['ms_level'] == 1) & (gpu_df['Annotation'] == 'prec')]
                    ms1_gpu = ms1_gpu[['Peptide', 'Charge', 'Run', 'rt', 'precursor_mz', 'int']]
                    ms1_gpu = ms1_gpu.rename(columns={'rt': 'RetentionTime',
                                                      'precursor_mz': 'Precursor_mz',
                                                      'int': 'Intensity'})
                    ms1_list.append(ms1_gpu.to_pandas())
                    
                    # ----- Extract MS2 (Fragment) Data -----
                    ms2_gpu = gpu_df[gpu_df['ms_level'] == 2]
                    ms2_gpu = ms2_gpu[['Peptide', 'Charge', 'Run', 'rt', 'mz', 'precursor_mz', 'int']]
                    ms2_gpu = ms2_gpu.rename(columns={'rt': 'RetentionTime',
                                                      'mz': 'Fragment_mz',
                                                      'precursor_mz': 'Precursor_mz',
                                                      'int': 'Intensity'})
                    ms2_list.append(ms2_gpu.to_pandas())
            
            # Save current batch results as CSV files
            if ms1_list:
                batch_ms1_df = pd.concat(ms1_list, ignore_index=True)
                batch_ms1_filename = os.path.join(self.ms1_dir, f"ms1_features_batch_{batch_idx}.csv")
                batch_ms1_df.to_csv(batch_ms1_filename, index=False)
                ms1_all_batches.append(batch_ms1_df)
            if ms2_list:
                batch_ms2_df = pd.concat(ms2_list, ignore_index=True)
                batch_ms2_filename = os.path.join(self.ms2_dir, f"ms2_features_batch_{batch_idx}.csv")
                batch_ms2_df.to_csv(batch_ms2_filename, index=False)
                ms2_all_batches.append(batch_ms2_df)
            
            batch_elapsed = time.time() - batch_start_time
            logging.info(f"Finished processing batch {batch_idx+1} in {batch_elapsed:.2f} seconds.")
        
        # Combine all batch results and export final combined files.
        if ms1_all_batches:
            final_ms1_df = pd.concat(ms1_all_batches, ignore_index=True)
            final_ms1_file = os.path.join(self.ms1_dir, "ms1_features_all_peptides_combined.csv")
            final_ms1_df.to_csv(final_ms1_file, index=False)
        else:
            final_ms1_file = None
        if ms2_all_batches:
            final_ms2_df = pd.concat(ms2_all_batches, ignore_index=True)
            final_ms2_file = os.path.join(self.ms2_dir, "ms2_features_all_peptides_combined.csv")
            final_ms2_df.to_csv(final_ms2_file, index=False)
        else:
            final_ms2_file = None
        
        logging.info("All batches processed and combined.")
        summary = {
            "total_peptides": len(unique_peptides),
            "ms1_features_combined": final_ms1_file,
            "ms2_features_combined": final_ms2_file
        }
        return summary

    def run_pipeline(self):
        """
        Run the complete feature extraction pipeline.

        Steps:
          1. Convert DIA-NN parquet files to TSV.
          2. Initialize the MassDASH loader and configure extraction.
          3. Run batch processing to extract MS1 and MS2 features.
        
        Returns:
            dict: Summary information including number of peptides processed and file paths.
        """
        logging.info("Step 1: Converting DIA-NN parquet files to TSV format")
        self.convert_parquet_to_tsv()
        
        logging.info("Step 2: Initializing MassDASH loader and configuring extraction")
        self.initialize_loader()
        self.configure_extraction()
        
        logging.info("Step 3: Processing peptides in batches")
        summary = self.run_batch_processing()
        
        logging.info(f"Pipeline execution complete. Results saved to {self.output_dir}")
        return summary

def load_config(config_path):
    """
    Load JSON configuration from the specified file.

    Args:
        config_path (str): Path to the JSON configuration file.
    
    Returns:
        dict: Parsed configuration.
    """
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        return config
    except Exception as e:
        logging.error(f"Error reading config file {config_path}: {e}")
        exit(1)

def main():
    """
    Main entry point for the feature extraction script.

    Parses command-line arguments, loads the configuration,
    maps configuration settings to processor parameters,
    and runs the complete feature extraction pipeline.
    """
    parser = argparse.ArgumentParser(
        description="DIA-NN to MassDASH Feature Extractor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config',
        type=str,
        default="dquartic_train_config.json",
        help="Path to the JSON configuration file."
    )
    args = parser.parse_args()
    
    config = load_config(args.config)
    data_config = config.get("data", {})
    threads = config.get("threads", 4)
    
    # Map JSON configuration keys to processor parameters.
    report_path = data_config.get("ms1_data_path", "report.parquet")
    gen_lib_path = data_config.get("ms2_data_path", "gen_spec_lib.parquet")
    mzml_path = data_config.get("raw_files", "/path/to/mzML/files/*.mzML")
    
    processor = DIAtoMassProcessor(
        report_path=report_path,
        gen_lib_path=gen_lib_path,
        mzml_path=mzml_path,
        batch_size=2000,
        threads=threads,
        output_dir="output_features",
        rt_window=50,
        mz_tol=20
    )
    
    summary = processor.run_pipeline()
    
    # Print summary information and integration details
    print("\nProcessing Summary:")
    print(f"Total peptides processed: {summary['total_peptides']}")
    print(f"MS1 features combined file: {summary['ms1_features_combined']}")
    print(f"MS2 features combined file: {summary['ms2_features_combined']}")
    
    print("\nIntegration with diffusion-deconvolution-dia-msms-data:")
    print("Add the following to your dquartic_train_config.json:")
    print('{\n  "data": {\n'
          f'    "ms1_data_path": "{summary["ms1_features_combined"]}",\n'
          f'    "ms2_data_path": "{summary["ms2_features_combined"]}",\n'
          '    "normalize": "minmax"\n'
          '  }\n}')

if __name__ == "__main__":
    main()
