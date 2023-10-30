import torch
import torchvision
from torch.utils import data
from torchvision import transforms

# 数据处理
transform = transforms.ToTensor()
train_data = torchvision.datasets.FashionMNIST("../../dataset",train=True,transform=transform,download=True)
test_data = torchvision.datasets.FashionMNIST("../../dataset",transform=transform,train=False,download=True)
train_loader = data.DataLoader(train_data,batch_size=256)
train_loader = data.DataLoader(test_data,batch_size=18)
for x,y in train_data:
    print(x.shape)
    print(y)



