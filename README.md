# Semester Project LPSY Adrien Feillard

# Getting started
## Project description

## Overview

This project explores the preprocessing and analysis of EEG data with a focus on steady-state visual evoked potentials (SSVEPs) and related frequency-domain phenomena.
This repository implements a complete EEG data analysis workflow, supporting:

 - Preprocessing raw EEG signals.

 - Artifact removal using ICA.

 - Visualization of Power Spectral Density (PSD) and topographic maps.

 - Frequency-specific filtering using RESS.

 - Machine learning-based contrast threshold prediction using LDA and SVM.

## Data

The dataset contains 128-channels EEG recording of 4 participants performing a visual task. Datas are separated in 3 folder: behavior data, trained data. In the Trained Data folder the train files cointains the EEG recordings, within the same folder test data can be found.

## Experiment 

(To fill)

## Repository Structure
.
├── Data/                     # Raw EEG datasets
│   ├── Behavior
│   ├── TrainedData
├── Image/                    # Output images (PSD plots, ICA components, etc.)
│   ├── RESS_matrix           # S & R covariance matrices for each RESS filtering
│   ├── Threshold_prediction  # Accuracy in function of contrast plots
│   │   ├── Baseline          # Treshold prediction using Baseline/Stimulus datas
│   │   ├── Raw               # Threshold prediction using Frequency with raw datas
│   │   ├── RESS_ch_filter    # Threshold prediction using Frequency with channel filtering and RESS filtering on raw datas
│   │   ├── RESS_raw          # Threshold prediction using Frequency with no channel filtering and with RESS filtering on raw datas
│   ├── Px_train_ICA          # ICA related plots of train raw datas
│   ├── Px_test_ICA           # ICA related plots of test raw datas
│   ├── PSD_plots             # PSD plots of the raw datas and the filtered datas
│   ├── PSD_Topomap_plots     # Topomap of the psd values for each channels average over trials
├── file_handling.py          # Data loading and saving utilities
├── data_processing.py        # Preprocessing functions
├── utils.py                  # Utility functions
├── visualization.py          # Plotting and visualization
├── computation.py            # Baseline and Frequency classifiers
├── RESS_filter.py            # RESS pipeline implementation
├── main.ipynb                # Main pipeline script (Jupyter Notebook)
├── README.md                 # Project documentation

## Installation

Image folder is not necessary to download for the installation. However all the .py and .ipynb files aswell as the data folder are necessary to compute the file main.ipynb
### Prerequisites

- Python 3.8.19 and/or conda 23.1.0
- Required libraries: `mne`, `scipy`, `numpy`, `matplotlib`, `pandas`, `sklearn`, `seaborn`,`scikit-learn`, `jupyter`, `jupyterlab`

