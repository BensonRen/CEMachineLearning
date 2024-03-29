# Create the neural network module: LeNet-5
"""
Assignment 2(a)
Build the LeNet-5 model by following table 1 or figure 1.

You can also insert batch normalization and leave the LeNet-5
with batch normalization here for assignment 3(c).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self,trainloader, valloader,device):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device

    def forward(self, x):
        #out = F.relu(self.conv1(x))                     #This is without Batch Norm
        out = F.relu(self.bn1(self.conv1(x)))          #This is the one with Batch Norm
        out = F.max_pool2d(out, 2)
        #out = F.relu(self.conv2(out))                   #THis is without Batch Norm
        out = F.relu(self.bn2(self.conv2(out)))        #This is the one with Batch Norm
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
