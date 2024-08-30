import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train_model(model, train_loader, test_loader, epochs, lr, device, model_path, patience=3):
    
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=patience, verbose=True)
    
    early_stop_count = 0
    min_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in test_loader:
                x_batch, y_batch = batch
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                val_losses.append(loss.item())

        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            early_stop_count = 0
            if model_path:
                torch.save(model.state_dict(), model_path)

        else:
            early_stop_count += 1

        if early_stop_count >= patience:
            print("Early stopping!")
            break
        
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss:.4f}")

    return model