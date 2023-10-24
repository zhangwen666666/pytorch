import torch
from torchvision import transforms
from PIL import Image
from model import LeNet

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

model = LeNet()
model.load_state_dict(torch.load('./model/model.text'))

for i in range(1, 4):
    path = str(i) + '.jpg'
    im = Image.open(path)  # 载入图片
    im = transform(im)  # 进行预处理  [C,H,W]
    im = torch.unsqueeze(im, dim=0)  # 转换为[batch,C,H,W]

    with torch.no_grad():
        outputs = model(im)
        predict = torch.max(outputs,dim=1)[1]
        print(classes[int(predict)])
