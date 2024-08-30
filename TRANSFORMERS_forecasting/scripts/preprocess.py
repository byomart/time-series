import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

class SunspotDataProcessor:
    def __init__(self, sequence_size=10, batch_size=32):
        self.sequence_size = sequence_size
        self.batch_size = batch_size
        self.scaler = StandardScaler()
    
    def preprocess_and_generate_sequences(self, df):
        """
        Preprocess the data and generate sequences for training and testing.

        Args:
            df (pd.DataFrame): The input DataFrame containing the data.

        Returns:
            tuple: Processed train and test sequences, test dates.
        """
        # Filter the data
        start_id = max(df[df['obs_num'] == 0].index.tolist()) + 1
        df = df[start_id:].copy()
        df['sn_value'] = df['sn_value'].astype(float)
        df_train = df[df['year'] < 2000]
        df_test = df[df['year'] >= 2000]

        # Scale the data
        spots_train = df_train['sn_value'].to_numpy().reshape(-1, 1)
        spots_test = df_test['sn_value'].to_numpy().reshape(-1, 1)
        spots_train = self.scaler.fit_transform(spots_train).flatten().tolist()
        spots_test = self.scaler.transform(spots_test).flatten().tolist()

        # Generate sequences
        x_train, y_train = self._to_sequences(self.sequence_size, spots_train)
        x_test, y_test = self._to_sequences(self.sequence_size, spots_test)
        test_dates = df_test["dec_date"][self.sequence_size:]

        return x_train, y_train, x_test, y_test, test_dates

    def setup_data_loaders(self, x_train, y_train, x_test, y_test):
        """
        Setup the data loaders for training and testing.

        Args:
            x_train (torch.Tensor): Training data sequences.
            y_train (torch.Tensor): Training labels.
            x_test (torch.Tensor): Testing data sequences.
            y_test (torch.Tensor): Testing labels.

        Returns:
            tuple: Train and test DataLoaders.
        """
        # Setup data loaders for batch processing
        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        test_dataset = TensorDataset(x_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader

    def _to_sequences(self, seq_size, obs):
        """
        Helper function to convert data into sequences.

        Args:
            seq_size (int): The size of the sequence window.
            obs (list): The observations to convert.

        Returns:
            tuple: Sequences and corresponding labels as torch Tensors.
        """
        x, y = [], []
        for i in range(len(obs) - seq_size):
            window = obs[i:(i + seq_size)]
            after_window = obs[i + seq_size]
            x.append(window)
            y.append(after_window)
        return torch.tensor(x, dtype=torch.float32).view(-1, seq_size, 1), torch.tensor(y, dtype=torch.float32).view(-1, 1)
