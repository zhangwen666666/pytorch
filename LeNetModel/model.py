import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # input(3,32,32)   output(16,28,28)
        x = self.pool1(x)  # input(16,28,28)   output(16,14,14)
        x = F.relu(self.conv2(x))  # input(16,14,14)   output(32,10,10)
        x = self.pool2(x)  # input(32,10,10)   output(32,5,5)
        x = x.view(-1, 32 * 5 * 5)  # input(32,5,5)   output(32*5*5)
        x = F.relu(self.fc1(x))  # input(32*5*5)   output(512)
        x = F.relu(self.fc2(x))  # input(512)   output(128)
        x = F.relu(self.fc3(x))  # input(128)   output(32)
        x = self.fc4(x)
        return x


# 调试
# import torch
# x = torch.rand([4,3,32,32])
# model = LeNet()
# print(model)
# y = model(x)
