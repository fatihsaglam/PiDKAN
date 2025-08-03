import pandas as pd
import numpy as np
import kan
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PiDKAN(nn.Module):
    def __init__(
            self, 
            input_size, 
            num_synaptic, 
            num_dendrite, 
            output_size, 
            # d, 
            # lambda_soma,
            grid,
            k):
        super(PiDKAN, self).__init__()

        self.KANSynapses = nn.ModuleList(nn.ModuleList(kan.KANLayer(in_dim = input_size, out_dim = 1, num = grid, k = k) for _ in range(num_synaptic)) for _ in range(num_dendrite))
        self.Sigmoid = nn.Sigmoid()
        self.FC = kan.KANLayer(in_dim = num_dendrite, out_dim = output_size, num = grid, k = k)
        self.d = nn.Parameter(torch.rand(1))
        self.lambda_soma = nn.Parameter(torch.rand(1))
        self.theta_soma = nn.Parameter(torch.rand(1))
        # self.d = d
        # self.lambda_soma = lambda_soma

    def forward(self, x):
        out_KS_all = [torch.cat([self.Sigmoid(self.d*synapse(x)[0]) for synapse in Dendrit], dim = 1) for Dendrit in self.KANSynapses]
        out_KD_all = [torch.prod(KS, dim = 1).reshape(x.shape[0],1) for KS in out_KS_all]
        # out_KM = self.FC(torch.cat(out_KD_all, dim = 1))[0]
        out_KM = torch.sum(torch.cat(out_KD_all, dim = 1), dim = 1)
        out_KO = self.Sigmoid(self.lambda_soma*(out_KM - self.theta_soma)).reshape(x.shape[0],1)
        return out_KO

def train(
        model, 
        train_loader, 
        val_loader=None, 
        num_epochs=50, 
        lr=0.001, 
        optimizer_type="adam", 
        early_stopping=True, 
        patience=5, verbose = True, scaler = None):
    
    model.to(device)
    criterion = nn.MSELoss()  # For regression (use CrossEntropyLoss for classification)

    if optimizer_type.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_type.lower() == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.75)
    # swa_model = AveragedModel(model)
    # scheduler = SWALR(optimizer, swa_lr=0.05)
    # swa_start = int(num_epochs * 0.75)

    best_val_loss = float("inf")
    no_improve_epochs = 0
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        if scaler:
            train_err = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs + 0.005, targets)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if scaler:
                train_err += np.sqrt(np.mean(np.pow((scaler.inverse_transform(targets) - scaler.inverse_transform(outputs.detach().numpy())),2)))

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation step
        val_loss = None
        if val_loader:
            model.eval()
            total_val_loss = 0
            if scaler:
                val_err = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    total_val_loss += criterion(outputs, targets).item()
                    
                    # if epoch < swa_start:
                    #     scheduler.step(val_loss)
                    # else:
                    #     # Start using SWA
                    #     swa_model.update_parameters(model)
                    #     scheduler.step()
                    
                    
                    prev_lr = scheduler.optimizer.param_groups[0]['lr']
                    scheduler.step(total_val_loss)
                    new_lr = scheduler.optimizer.param_groups[0]['lr']
                    if new_lr != prev_lr:
                        print(f"🔻 Learning rate reduced: {prev_lr:.6f} → {new_lr:.6f}")
                    
                    if scaler:
                        val_err += np.sqrt(np.mean(np.pow((scaler.inverse_transform(targets) - scaler.inverse_transform(outputs.detach().numpy())),2)))

            val_loss = total_val_loss / len(val_loader)
            if scaler:
                val_err = val_err / len(val_loader)
            val_losses.append(val_loss)

            if verbose:
                if (epoch + 1) % 10 == 0:
                    if scaler: 
                        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_err:.4f}, Val Loss: {val_err:.4f}")
                    else:
                        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Early stopping
            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
                    if no_improve_epochs >= patience:
                        if verbose:
                            print(f"Early stopping triggered at epoch {epoch+1}.")
                        break
        else:
            if verbose: 
                if (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch {epoch+1}/{num_epochs}, Train Loss: {np.sqrt(avg_train_loss):.4f}")

     
    return model, train_losses, val_losses

