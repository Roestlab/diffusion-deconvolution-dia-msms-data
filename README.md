# diffusion-deconvolution-dia-msms-data (D<sup>4</sup>)

## 🚀 Winner of the Donnelly Centre Innovation and Commercialization award!

Apply diffusion models to deconvolute highly multiplexed DIA-MS/MS data by conditioning on MS1 signals to generate cleaner MS2 data for downstream analysis.

![](https://img.shields.io/badge/License-BSD--3--Clause-blue?style=for-the-badge)

## Abstract

# Diffusion Deconvolution of DIA-MS/MS Data

As biological analysis machines and methodologies become more sophisticated and capable of handling more complex samples, the data they output also become more complicated to analyze. Modern generative machine learning techniques such as diffusion and score-based modeling have been used with great success in the domains of image, video, text, and audio data.

We aim to apply the same principles to highly multiplexed biological data signals and leverage the ability of generative models to learn the underlying distribution of the data, instead of just the boundaries using discriminative methods. We hope to apply diffusion models to signal denoising, specifically the deconvolution of highly multiplexed DIA-MS/MS data.

**DIA-MS/MS** features two types of data: MS1 and MS2. In MS1 data, information such as mass-to-charge ratio and chromatography elution time are recorded for entire peptides as they are analyzed. In MS2 data, the same information is recorded for the set MS2 peptide fragments belonging to the MS1 peptides onto the same data map. This means that although the data between MS1 and MS2 are correlated, the MS2 data can be highly multiplexed with signals from multiple MS1 peptides showing up.

Our project aims to train a diffusion model and condition it on MS1 data to deconvolute the corresponding MS2 signal, effectively simulating the case where the MS1 scan captured fewer peptides in its analysis window, producing cleaner MS2 data. This would be extremely useful for downstream analysis, identification, and quantification tasks.

We currently have access to a set of clean MS2 data which we plan to use to generate synthetic multiplexed MS2 data, and to use the corresponding clean MS1 data as a conditioning factor to re-extract the clean MS2. This should be an effective proof of concept for diffusion-based denoising of biological signal data.

## Installation

### 1. Clone the Repository

First, clone the repository to your local machine:

```bash
git clone git@github.com:hackbio-ca/diffusion-deconvolution-dia-msms.git
cd diffusion-deconvolution-dia-msms
```

### 2. Set Up the Environment

It's recommended to use a virtual environment. You can create one using Python's built-in venv module:

```bash
virtualenv dquartic
source dquartic/bin/activate  
```

Alternatively, you can use conda:

```bash
conda create -n dquartic python=3.9
conda activate dquartic
```

### 3. Install the library

To install the library and its dependencies, run:

```bash
pip install .
```

## Quick Start

The library has a CLI for training the diffusion model. Most configurations can be set via the [dquartic_train_config.json config file](https://github.com/Roestlab/diffusion-deconvolution-dia-msms-data/blob/main/dquartic_train_config.json)

```bash
$ dquartic train --help
Usage: dquartic train [OPTIONS] CONFIG_PATH

  Train a DDIM model on the DIAMS dataset.

Options:
  --parquet_directory TEXT  Path to the directory containing the Parquet files
                            containing the MS1 and MS2 data. Mutually
                            exclusive with `ms2_file` and `ms1_file`. Overides
                            config file
  --ms2-data-path TEXT      Path to MS2 data, overides config file
  --ms1-data-path TEXT      Path to MS1 data, overides config file
  --batch-size TEXT         Batch size for training, overides config file
  --checkpoint-path TEXT    Path to save the best model, overides config file
  --use-wandb TEXT          Use wandb for logging, overides config file
  --threads TEXT            Number of threads for data loading, overides
                            config file
  --help                    Show this message and exit.
```

## Usage

There is an example bash script for running a training example, which can be subitted via SLURM job

```bash
sbatch --job-name=myjob \
       --output=myjob_%j.out \
       --error=myjob_%j.err \
       --time=10:00:00 \
       --ntasks=1 \
       --gres=gpu:1 \
       --cpus-per-task=4 \
       --mem=16G \
       run_trainer.sh
```

## Contribute

Contributions are welcome! If you'd like to contribute, please open an issue or submit a pull request. See the [contribution guidelines](CONTRIBUTING.md) for more information.

## Support

If you have any issues or need help, please open an [issue](https://github.com/hackbio-ca/diffusion-deconvolution-dia-msms/issues) or contact the project maintainers.

## License

This project is licensed under the [BSD-3 License](LICENSE).
