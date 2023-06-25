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
+ matplotlib == 3.2.1

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

The training script is provided by `train.py` for the flight trajectory prediction. The arguments for the training process are defined bellow:

`--minibatch_len`: Integer. The sliding-window length for constructing the samples. `default=10`

`--interval`: Integer. The sampling period. `default=1`

`--batch_size`: Integer. The number of samples in a single training batch. `default=2048`

`--epoch`: Integer. The maximum epoch for training process. `default=150`

`--lr`: Float. The learning rate of the Adam optimizer. `default=0.001`

`--dpot`: Float. The dropout probability. `default=0.0`

`--cpu`: Optional. Use the CPU for training process.

`--nolongging`: Optional. The logs will not be recorded.

`--logdir`: String. The path for logs. `default='./log'`

`--datadir`: String. The path for dataset. `default='./data'`

`--saving_model_num`: Integer. The number of models to be saved during the training process. `default=0`

`--debug`: Optional. For debugging the scripts.

`--bidirectional`: Optional. Use the bidirectional LSTM block.

`--maxlevel`: Integer. The level of wavelet analysis. `default=1`

`--wavelet`: String. The wavelet basis. `default=haar`

`--wt_mode`: String. The signal extension mode for wavelet transform. `default=symmetric`

`--w_lo`: Float. The weight for the low-frequency wavelet component in the loss function. `default=1.0`

`--w_hi`: Float. The weight for the high-frequency wavelet components in the loss function. `default=1.0`

`--enlayer`: Integer. The layer number of the LSTM block in the encoder. `default=4`

`--delayer`: Integer. The layer number of the LSTM block in the decoder. `default=1`

`--embding`: Integer. The dimension of trajectory embeddings, enhanced trajectory embeddings and contextual embeddings. `default=64`

`--attn`: Optional. Use the wavelet attention module in the decoder.

`--cuda`: Integer. The GPU index for training process.

To train the WTFTP, use the following command.

```
python train.py --saving_model_num 10 --attn
```

To train the WTFTP without the wavelet attention module, use the following command.

```
python train.py --saving_model_num 10
```

To train the WTFTP of 2 or 3 level of wavelet analysis, use the following command.

```
python train.py --saving_model_num 10 --attn --maxlevel 2
``` 
```
python train.py --saving_model_num 10 --attn --maxlevel 3
```

To train the WTFTP without the wavelet attention module of 2 or 3 level of wavelet analysis, use the following command.

```
python train.py --saving_model_num 10 --maxlevel 2
``` 

```
python train.py --saving_model_num 10 --maxlevel 3
```


### Test

The test script is provided by `infer.py` for the evaluation. The arguments for the test process are defined bellow:

`--pre_len`: Integer. The prediction horizons for evaluation. `default=1`

`--batch_size`: Integer. The number of samples in a single test batch. `default=2048`

`--cpu`: Optional. Use the CPU for training process.

`--logdir`: String. The path for logs. `default='./log'`

`--datadir`: String. The path for dataset. `default='./data'`

`--netdir`: String. The path for the model.

To test the model, use the following command.

```
python infer.py --netdir ./xxx.pt
```

# Dataset

In this repository, the example samples are provided for evaluation. They can be accessed in the `\data\test`.


# Acknowledgment

The PyTorch implementation of wavelet transform is utilized to support the procedure of the DWT and IDWT procedures in this work. Its repository can be accessed [here](https://github.com/fbcotter/pytorch_wavelets). Thank all contributors to this project.

# Contact

Zheng Zhang (zhaeng@stu.scu.edu.cn, musevr.ae@gmail.com)
