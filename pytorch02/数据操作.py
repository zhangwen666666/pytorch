import torch

# t = torch.randint(1, 100, size=(4, 4))
# print(t)
# print(t[1][1])
# print(t[1, 2])
# print(t[1, :])

A = torch.arange(20).reshape(5, 4)
print(A)
# print(A.sum(axis=0))
# print(A)
# print(A.sum(axis=[0, 1]))
print(A.cumsum(axis=0))