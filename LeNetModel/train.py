import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from model import LeNet
from torch import nn
import os

# 对图像进行预处理的方法
transform = transforms.Compose([
    transforms.ToTensor(),
    # 将一个PIL图像或者一个numpy数组转换为tensor
    # PIL和numpy的形状是(H × W × C)，H、W、C范围在[0,255]
    # 转换为tensor之后形状是(C × H × W)，范围是[0,1]
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    # 使用均值mean和标准差std对数据进行标准化
])

# 50000张测试图片
train_data = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
train_loader = DataLoader(batch_size=72, dataset=train_data, shuffle=True, num_workers=0)

# 10000张训练图片
test_data = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
test_loader = DataLoader(batch_size=10000, dataset=test_data, shuffle=False, num_workers=0)
test_data_iter = iter(test_loader)  # 将test_loader转换为可迭代对象
test_image, test_label = next(test_data_iter)  # 获取到10000张图片以及他们对应的标签

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# 实例化模型
model = LeNet()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
if os.path.exists('./model/model.text'):  # 将模型加载到CPU上
    model.load_state_dict(torch.load('./model/model.text'))
    optimizer.load_state_dict(torch.load('./model/optimizer.text'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
model.to(device)  # 模型加载到GPU上
for state in optimizer.state.values():  # 将optimizer的tensor加载到GPU上
    for k, v in state.items():
        # print(k, v)
        if torch.is_tensor(v):
            state[k] = v.cuda()

# 训练模型
EPOCH = 20
for epoch in range(EPOCH):
    total_loss = 0.0
    total = 0
    for step, data in enumerate(train_loader, start=1):
        total += 1
        inputs, labels = data
        # print(labels.size(0))
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        total_loss += loss.item()  # 累加loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打印训练信息(测试)

    with torch.no_grad():
        test_image, test_label = test_image.to(device), test_label.to(device)
        outputs = model(test_image)  # [batch,10]
        predict_y = torch.max(outputs, dim=1)[1]
        accuracy = (predict_y == test_label).sum().item() / test_label.size(0)
        print('[%d/%d], loss = %.4f, accuracy = %.4f%%' % (epoch + 1,EPOCH, total_loss / total, accuracy * 100))
        total_loss = 0
        total = 0

print('Finished Training')
torch.save(model.state_dict(), './model/model.text')
torch.save(optimizer.state_dict(), './model/optimizer.text')
