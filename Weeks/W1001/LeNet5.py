#COS 470/570: LeNet5 implementation
#Xin Zhang

import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 1st convolutional layer
        self.conv1 = nn.Conv2d(1, 6, 5, 1)
        # 2nd convolutional layer
        self.conv2 = nn.Conv2d(6, 16, 5, 1)
        # FC
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  
        self.fc2 = nn.Linear(120, 84)
        # 10 categories
        self.fc3 = nn.Linear(84, 10)  

    def forward(self, x):
        # Convolution 1
        x = F.relu(self.conv1(x))
        # Pooling 1
        x = F.avg_pool2d(x, 2, 2)
        # Convolution 2
        x = F.relu(self.conv2(x))
        # Pooling 2
        x = F.avg_pool2d(x, 2, 2)
        # Flattening the output for the fully connected layer
        x = x.view(-1, 16 * 5 * 5)
        # Fully connected 1
        x = F.relu(self.fc1(x))
        # Fully connected 2
        x = F.relu(self.fc2(x))
        # Output layer
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

