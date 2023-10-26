import torch

t = torch.randn(3, 4, 5, 6)
print(t.shape)  # torch.Size([3, 4, 5, 6])

t = torch.flatten(t)
print(t.shape)  # torch.Size([360])
