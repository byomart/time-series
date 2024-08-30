import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import logging
import os
import torch
import torch.nn as nn


# Codificacion Posicional
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        logging.info(f'matriz de ceros: {pe.shape}')
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        logging.info(f'vector numeros hasta max_len (len sequencia): {position.shape}')
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        logging.info(f'factores por los que se multiplica: {div_term.shape}')
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        logging.info(f'matriz de codificaciones posicionales: {pe.shape}')
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :] #Añade las codificaciones posicionales pe al tensor de entrada x
        return self.dropout(x)
    



# Model definition using Transformer
class TransformerModel(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_layers=2, dropout=0.2):
        super(TransformerModel, self).__init__()

        self.encoder = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.decoder(x[:, -1, :])
        return x
    


def draw_predictions(zoom_data, dates, prediction, real, rmse):
    
    plt.figure(figsize=(12,10))
    plt.plot(dates[:zoom_data], prediction[:zoom_data], label="Predicted")
    plt.plot(dates[:zoom_data], real[:zoom_data], label="Real")
    plt.legend()
    plt.text(0.05, 0.95, f'Avg loss: {rmse:.4f}', transform=plt.gca().transAxes,
         fontsize=14, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

    # # save image
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # filename = f'transf_forecast_{timestamp}.png'
    
    filename = f'transf_forecast.png'
    output_dir = 'outputs/images'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.show()
    plt.close()

