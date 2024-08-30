# Time Series Prediction with Transformers

This repository contains a project focused on predicting time series data using Transformer models. Unlike traditional methods like LSTM, Transformers process sequences in parallel, which can significantly enhance the efficiency and effectiveness of capturing long-range dependencies in time series data.

## Overview

In this project, we implement Transformer-based models for time series prediction over daily total sunspot number
data. Transformers, known for their parallel processing capabilities, require additional mechanisms to encode positional information in sequences since they do not inherently understand the order of tokens. To address this, we utilize Positional Encoding to provide the model with information about the position of each token in the sequence.

## Key Features

- **Transformer Architecture**: Utilizes the Transformer model for efficient and scalable time series forecasting.
- **Positional Encoding**: Implements Positional Encoding to incorporate information about the sequence's token positions, essential for capturing temporal dependencies in data processed in parallel.
- **Data Preprocessing**: Includes data normalization and sequence generation to prepare time series data for model training.
- **Model Training and Evaluation**: Includes functions for training the model with early stopping, validation, and performance evaluation.


## Model architecture

<p align="center">
<img width="1000" alt="Model Architecture" src="https://github.com/fbayomartinez/time-series/blob/14023f714e671aeec2d13bcf3afcf2c37ae0cb21/TRANSFORMERS_forecasting/outputs/images/architecture.png">
</p>


## Results
<p align="center">
<img width="1000" alt="Best sun spot prediction" src="https://github.com/fbayomartinez/time-series/blob/8ac4beb31584a3ec2f6730b24842d9d7d53bcc4f/TRANSFORMERS_forecasting/outputs/images/transf_forecast.png">
</p>

