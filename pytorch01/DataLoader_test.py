import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

transform = torchvision.transforms.ToTensor()
test_data = torchvision.datasets.CIFAR10("../../dataset", train=False, download=False, transform=transform)
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
writer = SummaryWriter("dataloader")

# 取出DataLoader中的每一个元素
for i, data in enumerate(test_loader):
    imgs, targets = data
    # print(imgs.shape)
    # print(targets)
    writer.add_images("data_loader", imgs, global_step=i)
writer.close()


"""
test_data[0]会调用Dataset的__getitem__()方法，该方法返回一个元组(img, target)
img0, target0 = test_data[0]
img1, target1 = test_data[1]
img2, target2 = test_data[2]
img3, target3 = test_data[3]
DataLoader会将img0，img1，img2，img3打包为imgs
将target0，target1，target,2，target3打包为targets
将imgs，targets做为一个元组返回，该元组就是DataLoader的一个元素
"""