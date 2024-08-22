# Electricity Consumption Prediction

In this project we want to predict electricity consumption by using an advanced variant of Recurrent Neural Networks (RNN), the Long and Short Term Memory Models (LTSM), which have been designed to address the gradient fading problem and capture long term dependencies in time series and sequences.

In the log file we have been representing the datasets used as well as the model architecture or the dimensions of the output of each layer. 

In the lightning_logs folder are the hyperparameters used for each run. Similarly, each of these runs or 'versions' is mapped to an image contained in the consumption_prediction folder, which indicates the actual value along with the prediction, as well as the average value of the model loss.

The most relevant parameters of each run are shown below:

- version 0: 1 epoch
    - Input: [32, 10, 1]
    - After LTSM layer: [32, 10, 64]
    - After last sequence selection + dropout: [32, 64]
    - After fc1: [32, 32]
    - After fc2: [32, 1]
- version 1: 3 epochs
- version 2: 5 epochs  (keep 5 epochs from here)
- version 3: bidirectional LTSM with:
    - Input: [32, 10, 1]
    - After LTSM layer: [32, 10, 128]
    - After last sequence selection + dropout: [32, 128]
    - After fc1: [128, 64]
    - After fc2: [64, 32]
    - After fc3: [32, 1]
- version 4: 3 LTSM layers (until now just 1 layer)
- version 5: 5 LTSM layers (BEST RESULT)

<p align="center">
<img width="750" alt="Best energy consumption prediction" src="[https://github.com/fbayomartinez/BrainBoxes-switch/assets/163590683/0c08e4ed-5f44-4d2b-94a9-54986a4f9094](https://github.com/fbayomartinez/time-series/blob/f85c5dd2d8727cbf154f008918697dab67d42c7d/electricity_consumption/consumption_prediction/electricity_consumption_prediction_20240822_121945_version5.png)">
</p>



- version 6: 10 LTSM layers (+ LOSS = OVERFITTING)
- version 7: 5 LTSM y lr=0.01 (before it was lr=0.0001)
- version 8: 5 LTSM y lr=0.001



