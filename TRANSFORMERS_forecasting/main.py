import train, test, utils
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
import logging
import yaml


logging.basicConfig(filename='logs/log.log', 
                    level='INFO')


# load config file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)


names = ['year', 'month', 'day', 'dec_date', 'sn_value',
         'sn_error', 'obs_num', 'unused1']
df = pd.read_csv("https://data.heatonresearch.com/data/t81-558/SN_d_tot_V2.0.csv",
    sep=';', header=None, names=names,
    na_values=['-1'], index_col=False)

logging.info(df.head())


# Data Preprocessing
start_id = max(df[df['obs_num'] == 0].index.tolist()) + 1
df = df[start_id:].copy()
df['sn_value'] = df['sn_value'].astype(float)
df_train = df[df['year'] < 2000]
df_test = df[df['year'] >= 2000]

# daily number of sunspots (list)
spots_train = df_train['sn_value'].to_numpy().reshape(-1, 1)
spots_test = df_test['sn_value'].to_numpy().reshape(-1, 1)

scaler = StandardScaler()
spots_train = scaler.fit_transform(spots_train).flatten().tolist()
spots_test = scaler.transform(spots_test).flatten().tolist()

# break into sequences
SEQUENCE_SIZE = 10
testDates = df_test["dec_date"][SEQUENCE_SIZE:]
logging.info(f'test_dates: {testDates}')


def to_sequences(seq_size, obs):
    x = []
    y = []
    for i in range(len(obs) - seq_size):
        window = obs[i:(i + seq_size)]
        after_window = obs[i + seq_size]
        x.append(window)
        y.append(after_window)
    return torch.tensor(x, dtype=torch.float32).view(-1, seq_size, 1), torch.tensor(y, dtype=torch.float32).view(-1, 1)

x_train, y_train = to_sequences(SEQUENCE_SIZE, spots_train)
x_test, y_test = to_sequences(SEQUENCE_SIZE, spots_test)
# print(x_train.size()) # torch.Size([55150, 10, 1])

# Setup data loaders for batch
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)



# MODEL
# parameters
model_path = config['paths']['model']
epochs = config['model parameters']['epochs']
lr = config['model parameters']['lr']

# cuda if possible
device = "cuda" if torch.cuda.is_available() else "cpu"

# # train
# model = utils.TransformerModel()
# trained_model = train.train_model(model, train_loader, test_loader, epochs, lr, patience=5, device="cuda", model_path)

# # load trained model
model = utils.TransformerModel().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()


# # test
scaled_predictions, scaled_y_test, rmse = test.evaluate_model(model, test_loader, scaler, y_test, device)
logging.info(f"Score (RMSE): {rmse:.4f}")

# draw predictions
zoom = config['images']['zoom']
utils.draw_predictions(zoom, testDates, scaled_predictions, scaled_y_test)
