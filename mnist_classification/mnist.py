import torch
from torchvision import datasets
from torchvision import transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader

batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST(root='../dataset/mnist', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_data, shuffle=True, batch_size=batch_size)
test_data = datasets.MNIST(root='../dataset/mnist', train=False, download=True, transform=transform)
test_loader = DataLoader(dataset=test_data, shuffle=False, batch_size=batch_size)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_layer1 = torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3)
        self.conv_layer2 = torch.nn.Conv2d(in_channels=10, out_channels=15, kernel_size=3)
        self.conv_layer3 = torch.nn.Conv2d(in_channels=15, out_channels=30, kernel_size=1)
        self.pooling = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear1 = torch.nn.Linear(120, 64)
        self.linear2 = torch.nn.Linear(64, 32)
        self.linear3 = torch.nn.Linear(32, 10)

    def forward(self, x):
        x = F.relu(self.pooling(self.conv_layer1(x)))
        x = F.relu(self.pooling(self.conv_layer2(x)))
        x = F.relu(self.pooling(self.conv_layer3(x)))
        x = x.view(-1, 120)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)


model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    total_loss = 0
    total = 0
    for i, data in enumerate(train_loader, 1):
        total += 1
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("[第%d轮]\tloss = %.4f" % (epoch + 1, total_loss / total))
    total_loss = 0
    total = 0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            total += labels.size(0)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            correct += (predicted == labels).sum().item()
    print('Accuracy on test set: %d %%' % (100 * correct / total))


import time

if __name__ == '__main__':
    start = time.time()
    for epoch in range(10):
        train(epoch)
        test()
    end = time.time()
    print("训练此网络共用时%ds" % (end - start))
