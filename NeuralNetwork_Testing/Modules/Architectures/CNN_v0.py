import torch
from torch import nn


class CNN_v0(nn.Module):
    def __init__(self, k1=3,k2=3,k3=3):
        super(CNN_v0, self).__init__()

        self.conv1 = nn.Conv1d(190, 256, kernel_size=k1, stride=1, padding=k1//2)
        self.batchnorm1 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(256, 256, kernel_size=k2, stride=1, padding=k2//2)
        self.batchnorm2 = nn.BatchNorm1d(256)

        self.conv3 = nn.Conv1d(256, 64, kernel_size=k3, stride=1, padding=k3//2)
        self.batchnorm3 = nn.BatchNorm1d(64)

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*40, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.ReLU(),
        )

    def f(self, x):
            x = self.conv1(x)
            x = self.batchnorm1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.batchnorm2(x)
            x = self.relu(x)
            x = self.conv3(x)
            x = self.batchnorm3(x)
            x = self.relu(x)
            x = self.fc1(x)
            return x

    def forward(self, x):
        try:
            x = self.f(x)
        except RuntimeError:
            #x = x.reshape(x.shape[0], 40, 190) #just for grid search
            x = torch.transpose(x, 1, 2)
            x = self.f(x)
        return x