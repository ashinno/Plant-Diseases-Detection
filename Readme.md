# Tomato leaf Disease Detection with Vision Transformer (ViT)

This project aims to develop a tomato leaf disease detection system using a Vision Transformer (ViT) model. The system classifies plant diseases based on images using a custom deep learning model built on top of a pre-trained Vision Transformer.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Validation](#validation)
- [Evaluation](#evaluation)
- [Results](#results)
- [Conclusion](#conclusion)

## Introduction

Plant diseases can significantly impact agricultural productivity. Early detection and diagnosis are crucial for effective management. This project uses a Vision Transformer (ViT) model to classify plant diseases from images, leveraging state-of-the-art deep learning techniques for high accuracy.

## Installation

Ensure you have the following dependencies installed:

- Python 3.7+
- PyTorch
- torchvision
- transformers
- scikit-learn
- matplotlib
- seaborn

You can install the necessary packages using pip:

```bash
pip install torch torchvision transformers scikit-learn matplotlib seaborn
```

## Dataset

The dataset used in this project is the "New Plant Diseases Dataset (Augmented)". It contains images of various plant diseases across different categories.

## Data Preprocessing

The images are preprocessed with the following steps:

- **Training Data**:
  - Resize to 224x224 pixels
  - Random horizontal flip
  - Random rotation (up to 10 degrees)
  - Color jitter (brightness, contrast, saturation, hue)
  - Normalize using ImageNet mean and standard deviation

- **Validation Data**:
  - Resize to 224x224 pixels
  - Normalize using ImageNet mean and standard deviation

## Model Architecture

The model consists of a pre-trained Vision Transformer (ViT) base model with a custom fully connected (FC) layer for classification.

- **Vision Transformer (ViT)**:
  - Configuration: `google/vit-base-patch16-224`
  - Output hidden size: 768

- **Custom Model**:
  - Fully connected layer with 10 output classes (assuming 10 disease categories)

## Training

The training process involves fine-tuning the last few layers of the ViT model along with the custom fully connected layer. Key steps include:

- Loss function: CrossEntropyLoss
- Optimizer: Adam with differential learning rates for different parts of the model
- Number of epochs: 20

The model is trained and validated in each epoch, and the best model (with highest validation accuracy) is saved.

## Validation

Validation is performed at the end of each epoch to evaluate the model's performance on unseen data. Key metrics include validation loss and accuracy.

## Evaluation

After training, the best model is loaded and evaluated on the validation dataset. Evaluation metrics include:

- Confusion matrix
- Classification report (precision, recall, F1-score for each class)
- Overall accuracy

## Results

The results of the training process are visualized using plots for:

- Training and validation loss
- Training and validation accuracy

Additionally, a confusion matrix is plotted to analyze the model's performance across different classes.

## Conclusion

This project demonstrates the application of Vision Transformers for plant disease detection. The fine-tuned ViT model achieves high accuracy in classifying various plant diseases, providing a valuable tool for agricultural diagnostics.

---
