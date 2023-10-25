import torch

inputs = torch.Tensor([[1, 2, 0, 3, 1],
                       [0, 1, 2, 3, 1],
                       [1, 2, 1, 0, 0],
                       [5, 2, 3, 1, 1],
                       [2, 1, 0, 1, 1]])
inputs = torch.reshape(inputs, (1, 1, 5, 5))


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3,ceil_mode=False)

    def forward(self, inputs):
        return self.maxpool(inputs)


model = Model()
outputs = model(inputs)
print(outputs)