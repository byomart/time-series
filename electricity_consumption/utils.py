from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import logging

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
    def __init__(self):
        super(ElectricityLstm, self).__init__()

        self.lossFunction = nn.MSELoss()

        self.lstm = nn.LSTM(input_size=1, hidden_size=64, batch_first=True)
        self.droput = nn.Dropout(0.2)
        self.fc1 = nn.Linear(in_features=64, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=1)

    def forward(self, X):
        print(X.shape)
        out, _ = self.lstm(X)
        print(out.shape)
        out = out[:, -1, :]
        out = self.droput(out)
        out = self.fc1(out)
        out = self.fc2(out)

        
        return out


    def training_step(self, batch, batchIndex):
        X, y = batch
        out = self(X)
        loss = self.lossFunction(out, y)

        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)
    



