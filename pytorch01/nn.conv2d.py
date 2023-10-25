import torch
import torchvision
from torch.utils.data import DataLoader

transform = torchvision.transforms.ToTensor()
dataset = torchvision.datasets.CIFAR10(root="../../dataset", train=False,
                                       transform=transform, download=False)
dataloader = DataLoader(dataset=dataset, batch_size=64)

  
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3)

    def forward(self, x):
        return self.conv1(x)


model = Model()
print(model)

for data in dataloader:
    imgs, targets = data
    output = model(imgs)
    print(output.shape)  # torch.Size([64, 6, 30, 30])
