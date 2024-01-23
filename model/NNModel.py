import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.convolutaional_neural_network_layers  = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=12, kernel_size=3, padding=1, stride=1), # For grayscale input images
                nn.ReLU(), # Activation function after the first convolutional layer
                nn.MaxPool2d(kernel_size=2), # Reduces spatial dimensions
                nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, padding=1, stride=1), # Second convolutional layer
                nn.ReLU(), # Activation function after the second convolutional layer
                nn.MaxPool2d(kernel_size=2)  # Further reduces spatial dimensions
        )
        self.linear_layers  = nn.Sequential(
                nn.Linear(in_features=24*7*7, out_features=64), # Fully connected layer          
                nn.ReLU(),
                nn.Dropout(p=0.1), # Dropout to prevent overfitting
                nn.Linear(in_features=64, out_features=10) # Final output layer
        )

    def forward(self, x):
        x = self.convolutaional_neural_network_layers(x) # Pass through convolutional layers
        x = x.view(x.size(0), -1) # Flatten the output for the dense layers
        x = self.linear_layers(x) # Pass through dense layers
        x = F.log_softmax(x, dim=1) # Apply log softmax on the final output
        return x