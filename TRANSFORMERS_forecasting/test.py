import numpy as np
import torch

def evaluate_model(model, test_loader, scaler, y_test, device):
    """
    Evalúa el modelo en el conjunto de datos de prueba y calcula el RMSE.

    Args:
        model (nn.Module): El modelo a evaluar.
        test_loader (DataLoader): El DataLoader para el conjunto de datos de prueba.
        scaler (StandardScaler): El escalador utilizado para normalizar los datos.
        y_test (torch.Tensor): Las etiquetas verdaderas para el conjunto de datos de prueba.
        device (str): El dispositivo en el que se encuentra el modelo ('cuda' o 'cpu').

    Returns:
        tuple: Las predicciones escaladas y las etiquetas verdaderas escaladas.
    """
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            x_batch, y_batch = batch
            x_batch = x_batch.to(device)
            outputs = model(x_batch)
            predictions.extend(outputs.squeeze().tolist())

    # Convertir predicciones a array de numpy
    predictions = np.array(predictions).reshape(-1, 1)
    y_test = y_test.numpy().reshape(-1, 1)

    # Invertir la normalización
    scaled_predictions = scaler.inverse_transform(predictions)
    scaled_y_test = scaler.inverse_transform(y_test)

    # Calcular RMSE
    rmse = np.sqrt(np.mean((scaled_predictions - scaled_y_test) ** 2))

    return scaled_predictions, scaled_y_test, rmse
