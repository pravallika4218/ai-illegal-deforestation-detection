# AI Illegal Deforestation Detection

## Overview
An AI-based Python application that detects illegal deforestation from multi-temporal satellite imagery using Convolutional Neural Networks (CNN).  
Includes a web interface using Flask for image upload and prediction.

## Tech Stack
- **Language:** Python  
- **Libraries:** TensorFlow, Keras, OpenCV, NumPy, Flask  
- **Model:** Custom CNN classifier  
- **Interface:** Flask web app

## Features
- Preprocesses satellite imagery (NDVI & normalization)
- Trains a CNN for forest vs deforested classification
- Web interface for uploading images and viewing results
- Prediction confidence and visual cues

## Dataset
The dataset includes multispectral satellite images labeled as forested or deforested.  
*Note:* Dataset too large to upload here → see link below.

👉 **Dataset link:** https://data.mendeley.com/datasets/59xmzmcsjz/1**

## How to Run
1. Clone the repo  
2. Create virtual environment:
```bash
pip install -r requirements.txt
