import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn

# 数据
trans = transforms.ToTensor()
train_data = datasets.FashionMNIST("../../datasets", train=True, transform=trans, download=False)
test_data = datasets.FashionMNIST("../../datasets", train=False, transform=trans, download=False)
train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
test_loader = DataLoader(test_data, batch_size=256, shuffle=False)
# print(train_data[0][1])

# 参数
num_inputs, num_outputs, num_hiddens = 784, 10, 256
W1 = nn.Parameter(torch.rand(num_inputs, num_hiddens, requires_grad=True))
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True))
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
params = [W1, b1, W2, b2]


# 激活函数
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(a, X)


# 模型
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X @ W1 + b1)
    return H @ W2 + b2


loss_fn = nn.CrossEntropyLoss()  # 损失函数
num_epoch, lr = 10, 0.1
optim = torch.optim.SGD(params, lr=lr)  # 优化器

 
# 训练
def train(epoch):
    running_loss = 0
    count = 0
    for data in train_loader:
        inputs, labels = data
        outputs = net(inputs)
        loss = loss_fn(outputs, labels)
        running_loss += loss.item()
        count += 1
        optim.zero_grad()
        loss.backward()
        optim.step()
    print(f"epoch{epoch},count = {count}, loss = {running_loss / count}")


if __name__ == '__main__':
    for i in range(3):
        train(i + 1)
