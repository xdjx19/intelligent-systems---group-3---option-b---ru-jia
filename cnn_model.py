# cnn_model.py

import os
import torch
import torch.nn as nn

class SmallCNN(nn.Module):  # Define a CNN model
    def __init__(self, num_classes: int = 10):  # Initialize the model
        super().__init__()
        # Define layers: two convolutional layers, max pooling, dropout, and two fully connected layers
        self.c1 = nn.Conv2d(1, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):  # Define the forward pass
        # Pass input through layers: conv -> relu -> pool (twice), dropout, flatten, and fully connected layers
        x = self.pool(torch.relu(self.c1(x)))
        x = self.pool(torch.relu(self.c2(x)))
        x = self.drop(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)  # Output predictions

def load_trained_model(weights_path="cnn_mnist.pt", device=None):  # Function to load the trained model
    """
    Loads model weights and returns (model, device).
    Raises FileNotFoundError if weights_path not found.
    """
    if device is None:  # Sets device for model
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SmallCNN().to(device)  # Start and move model to device
    if not os.path.exists(weights_path):  # Check for weight file
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    state = torch.load(weights_path, map_location=device)  # Load model state
    model.load_state_dict(state)  # Apply state to the model
    model.eval()  # Set model to evaluation mode
    return model, device  # Return loaded model and device
