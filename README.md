# Rock, Paper, Scissors Image Classification Project

This project involves building a neural network using TensorFlow to classify images of rock, paper, and scissors. It includes data preprocessing, image augmentation, and model training with an accuracy target of over 95%.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Libraries Used](#libraries-used)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Usage](#usage)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

## Introduction

This project is part of my journey in learning machine learning, focusing on image classification. The model is trained to identify images of rock, paper, and scissors, achieving an accuracy of over 95%.

## Dataset

The dataset used is sourced from the [Rock-Paper-Scissors dataset](https://github.com/dicodingacademy/assets/releases/download/release/rockpaperscissors.zip). It is divided into training and validation sets with a 60-40 split.

## Libraries Used

- **TensorFlow**: Used for building and training the neural network model.
  - `import tensorflow as tf`
- **Keras**: Used for high-level neural networks API, running on top of TensorFlow.
  - `from tensorflow.keras.preprocessing import image`
- **Matplotlib**: Used for plotting and visualizing images and model performance.
  - `import matplotlib.pyplot as plt`
  - `import matplotlib.image as mpimg`
- **PIL (Python Imaging Library)**: Used for image processing.
  - `from PIL import Image`
- **NumPy**: Used for numerical operations and handling image data.
  - `import numpy as np`
- **os**: Used for interacting with the operating system, such as handling file paths.
  - `import os`
- **zipfile**: Used for extracting the dataset from a zip file.
  - `import zipfile`
- **ImageDataGenerator**: Used for generating batches of tensor image data with real-time data augmentation.
  - `from tensorflow.keras.preprocessing.image import ImageDataGenerator`
- **EarlyStopping**: Used for stopping the training process once a specified metric reaches a certain value.
  - `from tensorflow.keras.callbacks import EarlyStopping`

## Model Architecture

The model is built using the Sequential API of Keras with the following layers:

- Convolutional layers with ReLU activation
- MaxPooling layers
- Dropout layers for regularization
- Dense layers with ReLU activation
- Output layer with softmax activation for classification

## Training

The model is trained with the following parameters:

- Optimizer: Adam
- Loss function: Categorical Crossentropy
- Metrics: Accuracy
- Epochs: 20
- Early stopping callback to stop training when accuracy reaches 95%

## Usage

To use this project, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/Kevinadiputra/rock-paper-scissors-classification.git
   cd rock-paper-scissors-classification
   ```

2. Download and extract the dataset:
   ```
   !wget https://github.com/dicodingacademy/assets/releases/download/release/rockpaperscissors.zip
   !unzip rockpaperscissors.zip -d /content/
   ```

3. Install the required libraries:
   ```
   pip install tensorflow matplotlib pillow numpy
   ```

4. Run the training script:
   ```
   python train_model.py
   ```

## Results

The model achieves an accuracy of over 95% on the validation set. Example predictions can be visualized in the `predictions.ipynb` notebook.
