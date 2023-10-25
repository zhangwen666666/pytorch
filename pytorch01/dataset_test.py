from torchvision import datasets
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

transform = transforms.ToTensor()
train_dataset = datasets.CIFAR10(root="../../dataset", train=True, download=False, transform=transform)
test_dataset = datasets.CIFAR10(root="../../dataset", train=False, download=False, transform=transform)
# print(train_dataset[0])
# print(train_dataset[0])
# print(train_dataset.classes)
writer = SummaryWriter("logs")
# writer.add_image("img", train_dataset[0][0], global_step=1)

for i in range(15):
    img = train_dataset[i][0]
    writer.add_image("train_data", img, global_step=i)

writer.close()
