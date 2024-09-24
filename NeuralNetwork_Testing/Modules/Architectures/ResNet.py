from torchvision.models import resnet18
import torch.nn as nn
import torch

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.resnet18 = resnet18().requires_grad_(False)
        self.resnet18.conv1 = nn.Conv2d(190, 64, kernel_size=(3, 3), stride=(1, 3), padding=(1, 3), bias=False).requires_grad_(True)
        self.resnet18.fc = nn.Linear(512, 1).requires_grad_(True)

    def preprocess(self, x):
        if len(x.shape) == 3:
            if x.shape[2] == 190:
                x = torch.transpose(x, 1, 2)
                x = x.unsqueeze(2)
                x = torch.cat(([x]*3), dim=2)
        return x
    def forward(self, x):
        x = self.preprocess(x)
        return self.resnet18(x)