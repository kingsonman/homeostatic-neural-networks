# Need is All You Need: Homeostatic Neural Networks Adapt to Concept Shift

## Overview

This repository contains code to implement studies from the paper "Need is All You Need: Homeostatic Neural Networks Adapt to Concept Shift." The paper introduces a homeostatic neural network that adjusts its learning rate based on the classification tasks it performs, enhancing adaptability under concept shift.

Read the full paper on arXiv: [Need is All You Need: Homeostatic Neural Networks Adapt to Concept Shift](https://arxiv.org/abs/2205.08645) By Kingson Man, Antonio Damasio, and Hartmut Neven.


## Files

- `mnist_utils.py`: Utility functions for handling the MNIST dataset.
- `lrr.py`: Implementation of a classifier with randomly regulated learning rate.
- `lrh.py`: Implementation of a homeostatic classifier with self-regulated learning rate.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/username/homeostatic-neural-networks.git
   cd homeostatic-neural-networks
   ```
2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### Running the Experiments

1. To train and evaluate the classifier with a randomly regulated learning rate:
   ```sh
   python lrr.py
   ```

2. To train and evaluate the homeostatic classifier:
   ```sh
   python lrh.py
   ```
