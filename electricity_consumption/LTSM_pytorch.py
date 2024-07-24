from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
import logging
import utils



import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset



logging.basicConfig(filename='logs/log.log',
                    level= 'INFO',
                    filemode='w')


###############################################################
# 1. DF (train + test)
###############################################################

# read and analyze the dataset
df = pd.read_csv("electricityConsumptionAndProductioction.csv")
utils.df_info(df)

# only interested in Date and Consumption columns
df = df[["DateTime", "Consumption"]]
window_size = 10
test_size = 0.1

# split train and test sets
train, test, testDates = utils.get_train_and_test(df, test_size, window_size)

# for a better performance with nn, we use StandardScaler() that standardizes features by removing the mean and scaling to unit variance
scaler = StandardScaler()
train = scaler.fit_transform(train).flatten().tolist()
test = scaler.transform(test).flatten().tolist()
# now we have zero mean and unit variance, so nn will learn more efficiently and effectively


# split datasets in sequences with a window size (10 data sequence)
xTrain, yTrain = utils.toSequence(train, window_size)
xTest, yTest = utils.toSequence(test, window_size)
logging.info(f' xTrain shape: {xTrain.shape}, yTrain shape: {yTrain.shape}')
logging.info(f' xTest shape: {xTest.shape}, yTest shape: {yTest.shape}')






###############################################################
# 2. PREDICTION MODEL
###############################################################

batch_size = 32
numberOfEpochs = 1


train_dataset = TensorDataset(xTrain, yTrain)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
test_dataset = TensorDataset(xTest, yTest)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
for batch in train_loader:
    X, y = batch
    logging.info(f' First Batch shape: X shape: {X.shape}, y shape: {y.shape}')
    break 

model = utils.ElectricityLstm()
logging.info(f' Model Architecture: {model}')


# callback to save checkpoints
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',  # Monitorizar la métrica de validación
    dirpath='my_checkpoints',  # Directorio para guardar checkpoints
    filename='best-checkpoint',  # Nombre del archivo
    save_top_k=1,  # Guardar solo el mejor modelo
    mode='min'  # Guardar el modelo con el menor valor de la métrica monitoreada
)

trainer = pl.Trainer(
    max_epochs=numberOfEpochs,
    callbacks=[checkpoint_callback]
)

# model.train()
trainer.fit(model, train_loader, test_loader)
# model.eval()



'''
predictions = []
actualLabels = []

for batch in test_loader:
    X, y = batch
    prediction = model(X)

    predictions.extend(prediction.detach().numpy().flatten())
    actualLabels.extend(y.detach().numpy().flatten())

scaledPredictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
scaledYTest = scaler.inverse_transform(np.array(actualLabels).reshape(-1, 1))

averageLoss = np.sqrt(np.mean((scaledPredictions - scaledYTest) ** 2))
print(f"averageLoss: {averageLoss}")

###############################################################
###############################################################


dataSetSize = 200

plt.figure(figsize=(12,10))
plt.plot(testDates[:dataSetSize], scaledPredictions[:dataSetSize], color= "pink")
plt.plot(testDates[:dataSetSize], scaledYTest[:dataSetSize], color= "green")
plt.legend(["Predicted", "Real"])
plt.xticks(np.arange(0, dataSetSize, round((dataSetSize / 5)/10)*10))
plt.xticks(rotation=45)
plt.show()

'''
