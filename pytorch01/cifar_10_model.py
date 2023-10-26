import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32,
                      kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


model = Model()
print(model)
inputs = torch.ones((64, 3, 32, 32))
output = model(inputs)
print(output.shape)  # torch.Size([64, 10])

writer = SummaryWriter("./log_seq")
writer.add_graph(model, inputs)
writer.close()
