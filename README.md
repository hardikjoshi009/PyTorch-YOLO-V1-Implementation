# PyTorch YOLO V1 Implementation

This project implements the YOLO (You Only Look Once) V1 object detection algorithm from scratch using PyTorch. The model was trained on the PascalVOC dataset, which contains annotated images for object detection. Key components such as grid cell prediction, bounding box regression, and class probability computation were implemented according to the original YOLO V1 architecture.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [YOLO V1 Algorithm](#yolo-v1-algorithm)
- [Results](#results)
- [Findings and Insights](#findings-and-insights)

## Overview

YOLO V1 is a real-time object detection system that divides the image into a grid and predicts bounding boxes and class probabilities directly from the image. This implementation follows the original architecture and methodology described in the [YOLO V1 paper](https://arxiv.org/abs/1506.02640).

The model was trained on the PascalVOC dataset for 5 epochs and achieved a mean Average Precision (mAP) of 54.62%.

## Features

- **YOLO V1 from scratch**: Fully implemented in PyTorch, including the grid cell-based approach to object detection.
- **Bounding Box Regression**: Predicts bounding box coordinates relative to each grid cell.
- **Class Probability Prediction**: Each grid cell predicts a probability distribution over object classes.
- **PascalVOC Dataset**: Model trained and evaluated on the PascalVOC dataset for object detection tasks.
- **Training Process**: Custom training loop designed to handle loss calculations for bounding boxes, class probabilities, and confidence scores.

## Model Architecture

The architecture of YOLO V1 consists of:
- **Convolutional Layers**: Extract spatial features from the input image.
- **Fully Connected Layers**: Predict bounding box coordinates, confidence scores, and class probabilities for each grid cell.
- **Output Structure**: For each grid cell, the model outputs bounding boxes, confidence scores, and class probabilities. The final prediction tensor has dimensions: `(S x S x (B * 5 + C))`, where:
  - `S`: Grid size (e.g., 7x7)
  - `B`: Number of bounding boxes per grid cell
  - `C`: Number of object classes

## YOLO V1 Algorithm

The YOLO V1 algorithm divides the input image into an `S x S` grid. Each grid cell predicts `B` bounding boxes and corresponding confidence scores, as well as the probability distribution over `C` classes.

Key components include:
- **Grid Cell Prediction**: The image is divided into a grid, and each cell predicts bounding boxes and class probabilities.
- **Bounding Box Regression**: Each grid cell predicts the location and dimensions (x, y, width, height) of `B` bounding boxes.
- **Class Probability Computation**: For each grid cell, the model predicts a probability distribution over all object classes.

## Results

The model was successfully trained on the PascalVOC dataset for 5 epochs, achieving a mean Average Precision (mAP) of 54.62%.

## Findings and Insights

### Findings
1. **Performance**:  
   - Achieved a mean Average Precision (mAP) of **54.62%**, which is consistent with the expected performance of YOLO V1 on the PascalVOC dataset.  
   - The model performed well on larger objects but struggled with small objects due to the limited spatial resolution of grid cells.

2. **Training Challenges**:  
   - Balancing the loss terms for bounding box regression, class probabilities, and confidence scores was critical for stable training.  
   - Overfitting was observed after 5 epochs, necessitating early stopping or regularization techniques.  

3. **Grid Cell Limitations**:  
   - Objects spanning multiple grid cells were sometimes misclassified or split into multiple detections, a known limitation of YOLO V1.  

### Insights
1. **Impact of Data Augmentation**:  
   - Techniques like random cropping, flipping, and color jittering significantly improved model generalization to unseen data.  

2. **Role of Bounding Box Priors**:  
   - Initializing bounding box priors closer to the dataset distribution helped the model converge faster.  

3. **Small Object Detection**:  
   - The fixed grid size limits the detection performance for small objects, highlighting the need for multi-scale feature extraction in later YOLO versions.  

4. **Real-Time Suitability**:  
   - The lightweight architecture of YOLO V1 ensures real-time inference capability, making it ideal for applications requiring fast detection.
