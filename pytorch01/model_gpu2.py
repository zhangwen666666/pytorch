import time
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

# 定义训练的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(torch.cuda.is_available())
print(device)

# 准备数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
train_data = torchvision.datasets.CIFAR10("../../dataset", train=True, transform=transform)
test_data = torchvision.datasets.CIFAR10("../../dataset", train=False, transform=transform)

# length 长度
train_data_len = len(train_data)
test_data_len = len(test_data)
print(f"训练数据集长度为{train_data_len}")
print(f"测试数据集长度为{test_data_len}")

# 数据加载
train_loader = DataLoader(train_data, shuffle=True, batch_size=64)
test_loader = DataLoader(test_data, shuffle=False, batch_size=64)


# 创建网络
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 5, 1, 2),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 32, 5, 1, 2),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 5, 1, 2),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 4 * 4, 64),
            torch.nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


model = Model()
model = model.to(device)

# 损失函数
loss_fn = torch.nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器
optim = torch.optim.SGD(model.parameters(), lr=0.01)

start_time = time.time()
for epoch in range(10):
    running_loss = 0
    for data in test_loader:
        imgs, labels = data
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = loss_fn(outputs, labels)
        optim.zero_grad()
        loss.backward()
        optim.step()
        running_loss += loss.item()
    print("第%d次训练：loss = %.4f" % (epoch + 1, running_loss / train_data_len))

    running_loss = 0
    with torch.no_grad():
        count = 0
        for data in test_loader:
            imgs, labels = data
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            count += (outputs.argmax(1) == labels).sum()
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()
        print("第%d次训练：loss = %.4f, 准确率为：%.4f%%" % (
            epoch + 1, running_loss / train_data_len, count / test_data_len * 100))

end_time = time.time()
print(f"训练共用时{end_time - start_time}")
