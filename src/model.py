import torch
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        
        # Print initial expected input size
        #!print print("Expected input size: [batch_size, 1, 128, 431]")
        
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            
            # Second conv block
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            
            # Third conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.4),
        )
        
        # Calculate the size of flattened features
        with torch.no_grad():
            # Create dummy input
            dummy_input = torch.randn(1, 1, 128, 431)
            # Pass through conv layers
            dummy_output = self.conv_layers(dummy_input)
            # Calculate flattened size
            self.flatten_features = dummy_output.numel() // dummy_output.size(0)
            
            #!print print(f"Flattened features size: {self.flatten_features}")
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flatten_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Add dimension checking
        if len(x.shape) != 4:
            raise ValueError(f"Expected 4D input (batch_size, channels, height, width), got {len(x.shape)}D")
        if x.shape[1] != 1:
            raise ValueError(f"Expected 1 channel, got {x.shape[1]}")
        if x.shape[2] != 128:
            raise ValueError(f"Expected height 128, got {x.shape[2]}")
        
        # Print input shape for debugging
        #!print print(f"Input shape: {x.shape}")
        
        x = self.conv_layers(x)
        # Print shape after conv layers
        #!print print(f"Shape after conv layers: {x.shape}")
        
        x = x.view(x.size(0), -1)
        # Print shape after flattening
        #!print print(f"Shape after flattening: {x.shape}")
        
        x = self.fc_layers(x)
        return x
