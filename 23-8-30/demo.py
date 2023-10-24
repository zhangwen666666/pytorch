import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

# print(torch.__version__)

# x = torch.empty(5, 3)
# print(x)
# x = x.new_ones(5, 3, dtype=torch.double)
# print(x)
# print(x.size())

# x = torch.rand(4, 3, requires_grad=True)
# print(x)
# b = torch.rand(4, 3)
# b.requires_grad = True
# print(b)
# t = x + b
# print(t)
# y = t.sum()
# print(y)
# y.backward()
# print(b.grad)
# print(x.grad)

# print(torch.cuda.is_available())

# # 准备数据  y = 3x + 0.8
# x = torch.rand([2000, 1]) * 50
# y_true = 4 * x + 7
#
# # 计算预测值y_hat
# w = torch.rand([1,1], requires_grad=True)
# b = torch.rand(1, requires_grad=True)
#
# # 计算损失
#
# # 更新参数
# epoch = 100
# lr = 0.001
# for i in range(epoch):
#     y_hat = torch.matmul(x, w) + b
#     loss = ((y_true - y_hat) ** 2).mean()
#     print("epoch = %d, w = %.4f, b = %.4f.loss = %.4f" % (i, w.item(), b.item(), loss.item()))
#     if w.grad is not None:
#         w.grad.data.zero_()
#         print('w.yes')
#     if b.grad is not None:
#         b.grad.data.zero_()
#         print('b.yes')
#
#     loss.backward()
#     w.data -= lr * w.grad
#     b.data -= lr * b.grad
#
#     y_hat = torch.matmul(x, w) + b
#     loss = ((y_true - y_hat) ** 2).mean()
#
# # 预测


# class LinearRegressionModel(torch.nn.Module):
#     def __init__(self):
#         super(LinearRegressionModel, self).__init__()
#         self.linear = torch.nn.Linear(1, 1)
#
#     def forward(self, x):
#         y_hat = self.linear(x)
#         return y_hat
#
#
# model = LinearRegressionModel()
# criterion = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.005,momentum=0.5)
#
# x = torch.rand([500, 1])
# y_true = 3 * x + 0.8
#
# for epoch in range(10000):
#     y_hat = model(x)
#     loss = criterion(y_hat, y_true)
#     print("epoch = %-5d, w = %.6f, b = %.6f, loss = %.6f"%(epoch, model.linear.weight.item(), model.linear.bias.item(),loss.item()))
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
# import os
#
# transforms_fn = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=(0.1307,), std=(0.3081,))
# ])
# BATCH_SIZE = 128
#
#
# # 准备数据集
# def get_dataloader(train=True):
#     dataset = datasets.MNIST(root='../dataset/mnist', download=True, train=train, transform=transforms_fn)
#     data_loader = DataLoader(dataset=dataset, shuffle=train, batch_size=BATCH_SIZE)
#     return data_loader
#
#
# class MyModel(torch.nn.Module):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         self.fc1 = torch.nn.Linear(28 * 28 * 1, 28)
#         self.fc2 = torch.nn.Linear(28, 10)
#
#     def forward(self, x):
#         x = x.view(-1, 28 * 28 * 1)
#         x = self.fc1(x)
#         x = torch.nn.functional.relu(x)
#         x = self.fc2(x)
#         return x
#
#
# model = MyModel()
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
# if os.path.exists('./model/model.text'):
#     model.load_state_dict(torch.load('./model/model.text'))
#     optimizer.load_state_dict(torch.load('./model/optimizer.text'))
#
#
# def train(epoch):
#     print(epoch)
#     train_dataloader = get_dataloader(True)
#     for i, data in enumerate(train_dataloader):
#         inputs, labels = data
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         if i % 300 == 0:
#             print('loss = %.4f' % (loss.item()))
#             torch.save(model.state_dict(), './model/model.text')
#             torch.save(optimizer.state_dict(), './model/optimizer.text')
#
#
# if __name__ == '__main__':
#     for i in range(10):
#         train(i)

# import jieba
# text = '深度学习（英语：deep learning）是机器学习的分支，是一种以人工神经网络为架构，对数据进行表征学习的算法'
# cuted = jieba.lcut(text)
# # print(cuted)
# # 三个词语为一组做为特征
# list_cuted = [cuted[i:i+3] for i in range(len(cuted)-1)]
# print(list_cuted)

# BATCH_SIZE = 1
# SEQ_LEN = 3
# INPUT_SIZE = 4
# HIDDEN_SIZE = 2
#
# cell = torch.nn.RNNCell(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE)
#
# dataset = torch.randn(SEQ_LEN, BATCH_SIZE, INPUT_SIZE)  # 输入数据集x
# hidden = torch.zeros(BATCH_SIZE, HIDDEN_SIZE)  # 隐层的输入h0
#
# for idx, inputs in enumerate(dataset):
#     print('=' * 20, idx, '=' * 20)
#     print(inputs)
#     print('Input size:', inputs.shape)
#     hidden = cell(inputs, hidden)
#     print('Hidden size:', hidden.shape)
#     print(hidden)

# batch_size = 1
# seq_len = 3
# input_size = 4
# hidden_size = 2
# num_layers = 1
#
# cell = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size,
#                     num_layers=num_layers)
#
# inputs = torch.randn(seq_len, batch_size, input_size)  # 输入数据集x
# hidden = torch.zeros(num_layers, batch_size, hidden_size)  # 隐层的输入h0
#
# out, hidden = cell(inputs, hidden)
#
# print('output:', out)
# print('output size:', out.shape)
# print('hidden:', hidden)
# print('hidden size:', hidden.shape)
