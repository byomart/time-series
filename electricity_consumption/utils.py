from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import logging
import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl



def df_info(df, ):

    logging.info(f' Dataframe shape is: {df.shape}')
    logging.info(f' First 10 rows:')
    logging.info(df.head(10))
    logging.info('----------------------------------------------------------------')

    plt.figure(figsize=(12, 6))
    plt.suptitle('Electricity') 
    # subplot 1: consumption
    plt.subplot(2, 1, 1)
    plt.plot(df['Consumption'], label='Consumption', color='r')
    plt.ylabel('MW')
    plt.legend()
    # subplot 2: production
    plt.subplot(2, 1, 2)
    plt.plot(df['Production'], label='Production', color='g')
    plt.ylabel('MW')
    plt.legend()

    plt.savefig('electricity_consumption_production.png')
    plt.close()



def get_train_and_test(df, test_size = 0.1, sequenceSize = 10):
    '''
    Splits the DataFrame into training and testing sets, removing the dates and converting the values to 2D NumPy arrays.
    Args:
    df (pandas.DataFrame): The input DataFrame containing 'DateTime' and 'Consumption' columns.
    test_size (float): The size of the test set as a fraction of the total data (default is 0.1).
    sequenceSize (int): The sequence size used for visualization (default is 10).
    Returns:
    tuple: A tuple containing:
        - train (numpy.ndarray): The training dataset with the 'Consumption' column as a 2D array.
        - test (numpy.ndarray): The testing dataset with the 'Consumption' column as a 2D array.
        - testDates (pandas.Series): The dates corresponding to the test data (used for visualization).
    '''

    # split df in train and test
    train, test = train_test_split(df, test_size=test_size, shuffle=False)
    logging.info(f' Train data length: {len(train)}')
    logging.info(f' Train data : {train}')
    logging.info('----------------------------------------------------------------')
    logging.info(f' Test data length: {len(test)}')
    logging.info(f' Test data : {test}')
    logging.info('----------------------------------------------------------------')


    # date is important for visualization but then we can remove it from our train and test sets
    testDates = test["DateTime"][sequenceSize:]
    train = train["Consumption"]
    test = test["Consumption"]

    # data to numpy array and 2D shape
    train = train.to_numpy().reshape(-1, 1)
    test = test.to_numpy().reshape(-1, 1)

    return train, test, testDates


def toSequence(data, sequenceSize):
    '''
    Splits data into overlapping sequences of a given length, where each sequence corresponds to a target value immediately following the sequence. 
    Args:
    data: A 1D array or list of time series data
    sequenceSize (int): The length of each sequence.
    Returns:
    tuple: A tuple containing:
        - X (torch.Tensor): A 3D tensor where each element is a sequence of length `sequenceSize`, with shape 
                            `(number_of_sequences, sequenceSize, 1)`. Each sequence is extracted from the original data.
        - y (torch.Tensor): A 2D tensor where each element is the target value following the corresponding sequence in `X`, 
                            with shape `(number_of_sequences, 1)`.
    '''
    X = []
    y = []

    for i in range(len(data) - sequenceSize):
        window = data[i: (i + sequenceSize)]
        target = data[i + sequenceSize]

        X.append(window)
        y.append(target)

    return torch.tensor(X, dtype=torch.float32).reshape(-1, sequenceSize, 1), torch.tensor(y, dtype=torch.float32).reshape(-1, 1)



class ElectricityLstm(pl.LightningModule):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, dropout=0.2, out_features_fc1=32, out_features_fc2=1):
        super(ElectricityLstm, self).__init__()
        
        self.save_hyperparameters()

        self.lossFunction = nn.MSELoss()        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(in_features=hidden_size, out_features=out_features_fc1)
        self.fc2 = nn.Linear(in_features=out_features_fc1, out_features=out_features_fc2)


    def forward(self, X):
        # print('A', X.shape)
        out, _ = self.lstm(X)
        # print('B', out.shape)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc1(out)
        # print('C', out.shape)
        out = self.fc2(out)
        # print('D', out.shape)
        return out


    def training_step(self, batch, batchIndex):
        X, y = batch
        out = self(X)
        loss = self.lossFunction(out, y)

        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)
        loss = self.lossFunction(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)
    

def draw_predictions(dataSetSize, dates, prediction, real, avg_loss):
    plt.figure(figsize=(12,10))
    plt.plot(dates[:dataSetSize], prediction[:dataSetSize], label="Predicted")
    plt.plot(dates[:dataSetSize], real[:dataSetSize], label="Real")

    x_pos = dataSetSize * 0.01 
    y_pos = np.min(real) + (np.max(real) - np.min(real)) * 0.05  
    plt.text(
        x_pos, y_pos, f'Avg. Loss: {avg_loss:.2f}',  
        fontsize=9,
        color='black',
        bbox=dict(facecolor='white', alpha=0.5),
        horizontalalignment='left', 
        verticalalignment='bottom'  
    )

    plt.legend()
    plt.xticks(np.arange(0, dataSetSize, round((dataSetSize / 5)/10)*10))
    plt.xticks(rotation=0)

    # save image
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'electricity_consumption_prediction_{timestamp}.png'
    output_dir = 'consumption_prediction'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()