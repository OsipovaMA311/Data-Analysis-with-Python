import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        # ВАШ КОД ЗДЕСЬ
        # определите слои сети

        self.conv1 = nn.Conv2d(3, 3, kernel_size=(5, 5))
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(3, 5, kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.flatten = nn.Flatten()

        # После сверток и пулингов размер изображения будет 5x5
        # Вычисление: (32-5+1)/2 = 14 -> (14-3+1)/2 = 6 -> 6x6
        # Но с padding=0 по умолчанию: (32-4)/2=14 -> (14-2)/2=6 -> 6x6
        # 6x6 с 5 каналами = 6*6*5 = 180
        self.fc1 = nn.Linear(5 * 6 * 6, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        # размерность х ~ [64, 3, 32, 32]

        # ВАШ КОД ЗДЕСЬ
        # реализуйте forward pass сети

        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        
        return x
        
def create_model():
    return ConvNet()
