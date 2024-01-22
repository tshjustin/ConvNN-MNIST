import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, stride=1) # First Convolution Layer 

        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=1)  # Second Convolution Layer

        self.conv2_drop = nn.Dropout2d() # Regularization 

        self.fc1 = nn.Linear(320, 50) # Flattening 
        self.fc2 = nn.Linear(50, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = x.view(-1, 320)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
