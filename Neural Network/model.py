import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import numpy as np

from torch.nn.modules import activation

class NeuralNetwork(nn.Module):
    def __init__(self, layers:List, activation, bias=True):
        super(NeuralNetwork,self).__init__()

        architecture = []
        for i in range(1, len(layers)):
            architecture.append(nn.Linear(layers[i-1],layers[i],bias=bias))
            ## 最后一层不加激活函数
            if i < len(layers)-1:
                if activation == 'relu':
                    architecture.append(nn.ReLU())
                elif activation == 'sigmoid':
                    architecture.append(nn.Sigmoid())
                elif activation == 'tanh':
                    architecture.append(nn.Tanh())
                else:
                    raise

        self.fc = nn.Sequential(*architecture)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc(x)
        yhat = self.softmax(x)

        return yhat


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.avepool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avepool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(400,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        
        self.bn1 = nn.BatchNorm2d(6)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm1d(120)
        self.bn4 = nn.BatchNorm1d(84)


    def forward(self, x, batch_norm=False):
        x = F.sigmoid(self.conv1(x))
        if batch_norm:
            x = self.bn1(x)
        x = self.avepool1(x)

        x = F.sigmoid(self.conv2(x))
        if batch_norm:
            x = self.bn2(x)
        x = self.avepool2(x)

        x = x.view(-1,400)
        x = F.sigmoid(self.fc1(x))
        if batch_norm:
            x = self.bn3(x)
        x = F.sigmoid(self.fc2(x))
        if batch_norm:
            x = self.bn4(x)

        x = F.softmax(self.fc3(x), dim=-1)

        return x

if __name__=='__main__':

    layers = [4,10,10,3]
    activation = 'relu'

    x = np.random.randn(150,4)
    x = torch.from_numpy(x).float()
    nn = NeuralNetwork(layers=layers, activation=activation)
    yhat = nn(x)
    print(yhat)