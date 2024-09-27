# diffusion-deconvolution-dia-msms

Apply diffusion models to deconvolute highly multiplexed DIA-MS/MS data by conditioning on MS1 signals to generate cleaner MS2 data for downstream analysis.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Abstract

# Diffusion Deconvolution of DIA-MS/MS Data

**Keywords**: Deep Learning, Mass Spec, Signal Processing, Diffusion, DIA  
**Looking for Teammates**: YES  
**Expected Team Size**: 3-5  
**Required Skills**: Python, R, Bash, HPC  
**Computational Background**: 7  
**Biological Background**: 5  

As biological analysis machines and methodologies become more sophisticated and capable of handling more complex samples, the data they output also become more complicated to analyze. Modern generative machine learning techniques such as diffusion and score-based modeling have been used with great success in the domains of image, video, text, and audio data.

We aim to apply the same principles to highly multiplexed biological data signals and leverage the ability of generative models to learn the underlying distribution of the data, instead of just the boundaries using discriminative methods. We hope to apply diffusion models to signal denoising, specifically the deconvolution of highly multiplexed DIA-MS/MS data.

**DIA-MS/MS** features two types of data: MS1 and MS2. In MS1 data, information such as mass-to-charge ratio and chromatography elution time are recorded for entire peptides as they are analyzed. In MS2 data, the same information is recorded for the set MS2 peptide fragments belonging to the MS1 peptides onto the same data map. This means that although the data between MS1 and MS2 are correlated, the MS2 data can be highly multiplexed with signals from multiple MS1 peptides showing up.

Our project aims to train a diffusion model and condition it on MS1 data to deconvolute the corresponding MS2 signal, effectively simulating the case where the MS1 scan captured fewer peptides in its analysis window, producing cleaner MS2 data. This would be extremely useful for downstream analysis, identification, and quantification tasks.

We currently have access to a set of clean MS2 data which we plan to use to generate synthetic multiplexed MS2 data, and to use the corresponding clean MS1 data as a conditioning factor to re-extract the clean MS2. This should be an effective proof of concept for diffusion-based denoising of biological signal data.

## Installation

Provide instructions on how to install and set up the project, such as installing dependencies and preparing the environment.

```bash
# Example command to install dependencies (Python)
pip install project-dependencies

# Example command to install dependencies (R)
install.packages("project-dependencies")
```

## Quick Start

Provide a basic usage example or minimal code snippet that demonstrates how to use the project.

```python
# Example usage (Python)
import my_project

demo = my_project.example_function()
print(demo)
```
```r
# Example usage (R)
library(my_project)

demo <- example_function()
print(demo)
```

## Usage

Add detailed information and examples on how to use the project, covering its major features and functions.

```python
# More usage examples (Python)
import my_project

demo = my_project.advanced_function(parameter1='value1')
print(demo)
```
```r
# More usage examples (R)
library(demoProject)

demo <- advanced_function(parameter1 = "value1")
print(demo)
```

## Contribute

Contributions are welcome! If you'd like to contribute, please open an issue or submit a pull request. See the [contribution guidelines](CONTRIBUTING.md) for more information.

## Support

If you have any issues or need help, please open an [issue](https://github.com/hackbio-ca/diffusion-deconvolution-dia-msms/issues) or contact the project maintainers.

## License

This project is licensed under the [MIT License](LICENSE).
