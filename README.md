# Flight Trajectory Prediction Using Wavelet Transform: A Time-Frequency Perspective

## Update time: June 25, 2023

# Introduction

This repository provides source codes of the proposed flight trajectory prediction framework, called WTFTP, and example samples for the paper "Flight Trajectory Prediction Using Wavelet Transform: A Time-Frequency Perspective". 


## Repository Structure
```
wtftp-model
│  dataloader.py (Load trajectory data from ./data)
│  data_ranges.npy (Provide data ranges for normalization and denormalization)
│  desired_results.xlsx (Excepted prediction result of example samples)
│  infer.py (Perform the prediction procedure)
│  LICENSE (LICENSE file)
│  model.py (The neural architecture corresponding to the WTFTP framework)
│  README.md (The current README file)
│  train.py (The training code)
│  utils.py (Tools for the project)
│  
├─data
│  │  README.md (README file for the dataset.)
│  │  
│  ├─dev (Archive for the validation data)
│  ├─test (Archive for the test data)
│  └─train (Archive for the training data)
└─pics
```

## Package Requirements

+ Python == 3.7.1
+ torch == 1.4.0+cu100
+ numpy == 1.18.5
+ tensorboard == 2.3.0
+ tensorboardX == 2.1
+ PyWavelets == 1.2.0

## System Requirements
+ Ubuntu 16.04 operating system
+ Intel(R) Xeon(TM) E5-2690@2.90GHz
+ 128G of memory
+ 8TB of hard disks
+ 8 $\times$ NVIDIA(R) GeForce RTX(TM) 2080 Ti 11G GPUs.


# Instructions
## Installation

### Clone this repository

```
git clone https://github.com/MusDev7/wtftp-model.git
```

### Create proper software and hardware environment

You are recommended to create a conda environment with the package requirements mentioned above, and conduct the training and test on the suggested system configurations.

### Training

### Test

# Dataset

In this repository, the example samples are provided for evaluation. They can be accessed in the `\data\test`.


# Acknowledgment

The PyTorch implementation of wavelet transform is utilized to support the procedure of the DWT and IDWT procedures in this work. Its repository can be accessed [here](https://github.com/fbcotter/pytorch_wavelets). Thank all contributors to this project.

# Contact

Zheng Zhang (zhaeng@stu.scu.edu.cn, musevr.ae@gmail.com)