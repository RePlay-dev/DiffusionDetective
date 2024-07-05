# DiffusionDetective

<p align="center">
  <img src="img/screenshot.png" alt="Screenshot of DiffusionDetective">
</p>

DiffusionDetective is a project developed as part of a bachelor thesis titled "Detection of Diffusion-Generated Images Extended With Explainable AI Techniques." The project implements and compares multiple approaches for detecting diffusion-generated images. The main image classifier is built using FastAI and is based on the ConvNeXt v2 architecture. Additional detection models incorporated in this project include 'ClipBased-SyntheticImageDetection' and 'DMimageDetection'. To provide insights into the decision-making process of the classifier, the project integrates several Explainable AI (XAI) techniques: RISE (Random Input Sampling for Explanation), AblationCAM, and SHAP (SHapley Additive exPlanations). This repository contains the implementation of these models and explainers, along with documentation and experimental results. The project aims to contribute to the ongoing research in AI-generated image detection and explainable AI in the context of image classification.

## Installation

To create the Python environment, [Conda](https://docs.anaconda.com/miniconda/miniconda-install/) needs to be installed. The environment can be created using either the `environment.yml` or `environment_amd.yml` file, depending on your system configuration.

```sh
# For systems with a NVIDIA GPU
conda env create -f environment.yml

# For systems with an AMD GPU
conda env create -f environment_amd.yml
```

## Datasets

The file `data/README.md` describes how the datasets can be downloaded. Please refer to this file for detailed instructions.

## Usage

### Running Detectors and Getting Explanations

To run detectors and get explanations, use the `main.py` script:

```sh
python main.py
```

### Web Interface

To start the web interface, run:

```sh
python web.py
```

### Benchmarking Detectors

To run a large number of inferences with the detectors to compare them, use the `detector_benchmark.py` script:

```sh
python detector_benchmark.py
```

### Generating Explanations

To generate explanations for all images in a folder using all detectors and explainers, run the `explainer_benchmark.py` script:

```sh
python explainer_benchmark.py
```

### Training the DiffusionDetective Model

To train the DiffusionDetective model, open the `training.ipynb` notebook in Jupyter. Ensure that the datasets have been downloaded as per the instructions in `data/README.md`.

```sh
jupyter notebook training.ipynb
```

## Special Thanks

Special thanks to the following projects, from which code was adapted for DiffusionDetective:

- [RISE](https://github.com/eclique/RISE)
- [ClipBased-SyntheticImageDetection](https://github.com/grip-unina/ClipBased-SyntheticImageDetection/)
- [DMimageDetection](https://github.com/grip-unina/DMimageDetection)
