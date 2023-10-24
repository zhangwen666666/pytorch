import torch
import os

# Input_size = 4
# Hidden_size = 4
# Batch_size = 1
#
# idx2char = ['e', 'h', 'l', 'o']
# x_data = [1, 0, 2, 2, 3]
# y_data = [3, 1, 2, 3, 2]
# # 构造独热矩阵
# one_hot_lookup = [[1, 0, 0, 0],
#                   [0, 1, 0, 0],
#                   [0, 0, 1, 0],
#                   [0, 0, 0, 1]]
# x_one_hot = [one_hot_lookup[x] for x in x_data]  # 构造输入
# inputs = torch.tensor(x_one_hot, dtype=torch.float32).view(-1, Batch_size, Input_size)  # 转换形状
# labels = torch.LongTensor(y_data).view(-1, 1)  # 转换形状
#
#
# # 构造模型
# class Model(torch.nn.Module):
#     def __init__(self, input_size, hidden_size, batch_size):
#         super(Model, self).__init__()
#         # self.num_layers = num_layers
#         self.batch_size = batch_size
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.rnncell = torch.nn.RNNCell(input_size=self.input_size,
#                                         hidden_size=self.hidden_size)
#
#     def forward(self, input, hidden):
#         hidden = self.rnncell(input, hidden)
#         return hidden
#
#     def init_hidden(self):
#         ''' 构造h0 '''
#         return torch.zeros(self.batch_size, self.hidden_size)
#
#
# net = Model(Input_size, Hidden_size, Batch_size)
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
# # 模型的加载
# if os.path.exists('./model/model.text'):
#     net.load_state_dict(torch.load('./model/model.text'))
#     optimizer.load_state_dict(torch.load('./model/optimizer.text'))
#
# for epoch in range(15):
#     loss = 0
#     optimizer.zero_grad()
#     hidden = net.init_hidden()  # 初始化h0
#     # print('Predicted string: ', end='')
#     for input, label in zip(inputs, labels):
#         print('input:', input.shape)
#         print('hidden:', hidden.shape)
#         print('label:', label.shape)
#         hidden = net(input, hidden)
#         # 每次得到的loss是一个字符的loss，需要把每个字符的loss加起来才是模型的loss
#         loss += criterion(hidden, label)
#         # hidden是一个四维的输出，表示属于四个分类的概率，取最大值的下标
#         _, idx = hidden.max(dim=1)
#         # print(idx2char[idx.item()], end='') #根据下标和词典，将该字符打印出来
#     loss.backward()
#     optimizer.step()
#     # print(', Epoch [%d/15] loss = %.4f' % (epoch + 1, loss.item()))
#     # 模型的保存
#     torch.save(net.state_dict(), './model/model.text')
#     torch.save(optimizer.state_dict(), './model/optimizer.text')

Input_size = 4
Hidden_size = 4
Num_layers = 1
Batch_size = 1
Seq_len = 5
# 准备词典
idx2char = ['e', 'h', 'l', 'o']
x_data = [1, 0, 2, 2, 3]
y_data = [3, 1, 2, 3, 2]
# 构造独热矩阵
one_hot_lookup = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]
x_one_hot = [one_hot_lookup[x] for x in x_data]  # 构造输入
inputs = torch.tensor(x_one_hot, dtype=torch.float32).view(-1, Batch_size, Input_size)  # 转换形状
labels = torch.LongTensor(y_data) # 这里label是一个一维的向量


# 构建模型
class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers=1):
        super(Model, self).__init__()
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = torch.nn.RNN(input_size=self.input_size,
                                hidden_size=self.hidden_size,
                                num_layers=num_layers)

    def forward(self, inputs):
        hidden = torch.zeros(self.num_layers,
                             self.batch_size,
                             self.hidden_size)  # 构造h0
        out, _ = self.rnn(inputs, hidden)
        return out.view(-1, self.hidden_size)


net = Model(Input_size, Hidden_size, Batch_size, Num_layers)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

for epoch in range(15):
    optimizer.zero_grad()
    outputs = net(inputs)
    # print(outputs.shape)
    # print(labels.shape)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    # print(outputs)
    _, idx = outputs.max(dim=1)
    # print(_)
    # print(idx)
    idx = idx.data.numpy()
    print('Predicted: ', ''.join([idx2char[x] for x in idx]), end='')
    print(', Epoch [%d/15] loss = %.4f' % (epoch + 1, loss.item()))
