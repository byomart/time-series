import numpy as np
import torch

def evaluate_model(model, test_loader, scaler, y_test, device):
    """
    Evaluates the model on the test dataset and calculates the RMSE.

    Args:
        model (nn.Module): The model to evaluate.
        test_loader (DataLoader): The DataLoader for the test dataset.
        scaler (StandardScaler): The scaler used to normalize the data.
        y_test (torch.Tensor): The true labels for the test dataset.
        device (str): The device where the model is located ('cuda' or 'cpu').

    Returns:
        tuple: The scaled predictions and the scaled true labels.
    """
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            x_batch, y_batch = batch
            x_batch = x_batch.to(device)
            outputs = model(x_batch)
            predictions.extend(outputs.squeeze().tolist())

    # Convert predictions to numpy array
    predictions = np.array(predictions).reshape(-1, 1)
    y_test = y_test.numpy().reshape(-1, 1)

    # Reverse normalization
    scaled_predictions = scaler.inverse_transform(predictions)
    scaled_y_test = scaler.inverse_transform(y_test)

    # Calculate RMSE
    rmse = np.sqrt(np.mean((scaled_predictions - scaled_y_test) ** 2))

    return scaled_predictions, scaled_y_test, rmse
