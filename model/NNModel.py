import torch.nn as nn

class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.convolutaional_neural_network_layers  = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=12, kernel_size=3, padding=1, stride=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, padding=1, stride=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
        )
        self.linear_layers  = nn.Sequential(
                nn.Linear(in_features=24*7*7, out_features=64),          
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(in_features=64, out_features=10) 
        )

    def forward(self, x):
        x = self.convolutaional_neural_network_layers(x)
        x = x.view(x.size(0), -1) 
        x = self.linear_layers(x)
        return x