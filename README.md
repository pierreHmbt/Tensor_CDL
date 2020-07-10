# T-ConvFISTA algorithm

Code of the T-ConvFISTA (TC-FISTA) algorithm used in "Tensor Convolutional Sparse Coding with Low-Rankactivations, an application to EEG analysis". In addition, we provide a pretrained model on the Electroencephalogram signals (EEG) of the paper.

## Container

##### -- Notebooks --
The reposity contains the code of TC-FISTA and two notebooks:

1) One for synthetic data
2) One to visualize the pretrained model of the EEG application of the paper

##### -- Data --
The folder 'data' contains all nescessary file to visualize the EEG:
1) Raw signals (64 Mo)
2) Components learnt with TC-FISTA

One example of atom with its activations

<img src="./outputs_eeg/atom_active_0_.png" alt="drawing" width="1000"/>

## Requirements

This code runs on Python >= 3.5. Set up environment with:
```
pip install -r requirements.txt
```

numpy 1.14.2
tensorly 0.4.3
mne 0.17.2
sporco 0.1.10
matplotlib 3.0.0
