# src/cnn_custom.py
"""
Module 3: Custom Convolutional Neural Network (CNN) for Spatial Traffic Analysis.
Demonstrates core CNN concepts: Conv2D, BatchNorm, ReLU, MaxPooling, and FC layers.
Maps spatial occupancy grid / visual features to 4-approach traffic queue densities.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


class CustomTrafficCNN(nn.Module):
    """
    Custom Convolutional Neural Network implementing Module 3 concepts:
    - 2D Convolutions with configurable kernel sizes and padding
    - Batch Normalization for stabilized deep feature learning
    - ReLU activations and Max-Pooling for spatial downsampling
    - Fully connected read-out layers for multi-lane density prediction
    """
    def __init__(self, in_channels=3, num_lanes=4):
        super().__init__()
        # Convolutional Block 1 (receptive field expansion)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64x64 -> 32x32

        # Convolutional Block 2 (hierarchical feature extraction)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32 -> 16x16

        # Convolutional Block 3 (high-level spatial patterns)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16 -> 8x8

        # Spatial pooling to fixed feature map
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))   # -> 128 x 4 x 4

        # Fully Connected Classification / Density Estimation Head
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_lanes),
            nn.ReLU()  # Queues are non-negative
        )

    def forward(self, x):
        # x: (batch, channels, height, width)
        out = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        out = self.pool2(self.relu2(self.bn2(self.conv2(out))))
        out = self.pool3(self.relu3(self.bn3(self.conv3(out))))
        out = self.adaptive_pool(out)
        densities = self.fc(out)
        return densities


class SpatialOccupancyDataset(Dataset):
    """
    Dataset of spatial intersection grids and corresponding ground-truth queue densities.
    Generates synthetic 64x64 3-channel spatial heatmaps from queue values.
    """
    def __init__(self, num_samples=1000):
        self.samples = []
        for _ in range(num_samples):
            # 4 lane queues: North, South, East, West
            queues = np.random.uniform(0.0, 25.0, size=4).astype(np.float32)
            # Create a 64x64 3-channel occupancy grid
            grid = np.zeros((3, 64, 64), dtype=np.float32)

            # Draw lane heatmaps proportional to queue lengths
            # North approach (top center)
            n_len = int(np.clip(queues[0] * 1.2, 0, 24))
            if n_len > 0:
                grid[0, 28 - n_len:28, 28:36] += 0.8
                grid[1, 28 - n_len:28, 28:36] += 0.2

            # South approach (bottom center)
            s_len = int(np.clip(queues[1] * 1.2, 0, 24))
            if s_len > 0:
                grid[0, 36:36 + s_len, 28:36] += 0.2
                grid[1, 36:36 + s_len, 28:36] += 0.8

            # East approach (right center)
            e_len = int(np.clip(queues[2] * 1.2, 0, 24))
            if e_len > 0:
                grid[1, 28:36, 36:36 + e_len] += 0.7
                grid[2, 28:36, 36:36 + e_len] += 0.7

            # West approach (left center)
            w_len = int(np.clip(queues[3] * 1.2, 0, 24))
            if w_len > 0:
                grid[2, 28:36, 28 - w_len:28] += 0.9

            # Add gentle background noise
            grid += np.random.normal(0, 0.05, grid.shape).astype(np.float32)
            grid = np.clip(grid, 0.0, 1.0)

            self.samples.append((torch.from_numpy(grid), torch.from_numpy(queues)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def train_custom_cnn(epochs=10, batch_size=32, lr=1e-3, device="cpu", save_path="models/custom_cnn.pt"):
    """
    Trains and validates CustomTrafficCNN on spatial traffic density heatmaps.
    Saves checkpoint to save_path.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    train_dataset = SpatialOccupancyDataset(num_samples=1200)
    val_dataset = SpatialOccupancyDataset(num_samples=300)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = CustomTrafficCNN(in_channels=3, num_lanes=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    metrics = {"train_loss": [], "val_loss": [], "val_mae": []}

    print(f"Training CustomTrafficCNN for {epochs} epochs on device: {device}")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)

        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        total_mae = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x)
                loss = criterion(preds, y)
                val_loss += loss.item() * len(x)
                total_mae += torch.abs(preds - y).sum().item()

        val_loss /= len(val_dataset)
        val_mae = total_mae / (len(val_dataset) * 4)

        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["val_mae"].append(val_mae)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

        if epoch % 2 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f} | Val MAE: {val_mae:.2f} veh")

    print(f"CustomTrafficCNN trained and saved to {save_path} (Best Val MSE: {best_val_loss:.4f})")
    return model, metrics


def load_custom_cnn(checkpoint_path="models/custom_cnn.pt", device="cpu"):
    """Loads a pretrained or trained CustomTrafficCNN instance."""
    model = CustomTrafficCNN(in_channels=3, num_lanes=4).to(device)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
    return model
