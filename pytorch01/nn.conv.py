import torch

# 输入
inputs = torch.tensor([[1, 2, 0, 3, 1],
                       [0, 1, 2, 3, 1],
                       [1, 2, 1, 0, 0],
                       [5, 2, 3, 1, 1],
                       [2, 1, 0, 1, 1]])
# 卷积核
kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

# 对输入和卷积核进行尺寸变化
inputs = torch.reshape(inputs, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))

outputs = torch.nn.functional.conv2d(inputs, kernel, stride=1)
print(outputs)
