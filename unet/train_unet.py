import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from unet import AttentionUNet

class NAMODataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        # Convert float64 inputs to float32 tensors
        self.inputs = torch.tensor(data['inputs'], dtype=torch.float32)
        # Add channel dimension to labels: (N, 20, 20) -> (N, 1, 20, 20)
        self.labels = torch.tensor(data['labels'], dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

def train():
    npz_path = str(Path(__file__).resolve().parent.parent / "namo_dataset.npz")
    dataset = NAMODataset(npz_path)
    
    # 80/20 train/validation split
    train_size = 1200
    val_size = 300
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    
    # Check for Apple Silicon native GPU (MPS)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device for training: {device}")
    
    model = AttentionUNet().to(device)
    criterion = nn.MSELoss() # Best for continuous risk values (0.9, 0.4, 0.1, 0.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 15
    print(f"\nStarting Attention U-Net training on {len(train_set)} samples...")
    print("="*60)
    
    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * x.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item() * x.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
    # Save the trained model weights
    save_path = str(Path(__file__).resolve().parent / "namo_unet.pth")
    torch.save(model.state_dict(), save_path)
    print("="*60)
    print(f"Training successfully completed! Saved weights to {save_path}\n")

if __name__ == "__main__":
    train()
