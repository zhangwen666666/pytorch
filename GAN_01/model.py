import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

# 对数据归一化[-1,1]
transform = transforms.Compose([
    transforms.ToTensor(),  # 0-1归一化，[C,H,W]
    transforms.Normalize(0.5, 0.5)
])
# 加载数据
train_dataset = torchvision.datasets.MNIST('../dataset/mnist', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)


# 定义生成器网络
# 输入是长度为100的噪声(正态分布随机数)  输出是(1,28,28)的图片
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28 * 28),
            nn.Tanh()
        )

    def forward(self, x):
        img = self.main(x)
        img = img.view(-1, 1, 28, 28)
        return img


# 定义判别器模型
# 输入是生成器输出的一张(1,28,28)图片  输出为二分类的概率值，输出使用sigmoid激活
# BCEloss计算交叉熵损失
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.main(x)
        return x


device = 'cuda' if torch.cuda.is_available() else 'cpu'
gen = Generator()
dis = Discriminator()
g_optim = optim.Adam(gen.parameters(), lr=0.0001)
d_optim = optim.Adam(dis.parameters(), lr=0.0001)
if os.path.exists('./model/gen.text'):  # 将模型加载到CPU上
    gen.load_state_dict(torch.load('./model/gen.text'))
    dis.load_state_dict(torch.load('./model/dis.text'))
    g_optim.load_state_dict(torch.load('./model/g_optim.text'))
    d_optim.load_state_dict(torch.load('./model/d_optim.text'))
gen.to(device)  # 模型加载到GPU上
dis.to(device)
for state in g_optim.state.values():  # 将g_optim的tensor加载到GPU上
    for k, v in state.items():
        # print(k, v)
        if torch.is_tensor(v):
            state[k] = v.cuda()
for state in d_optim.state.values():  # 将d_optim的tensor加载到GPU上
    for k, v in state.items():
        # print(k, v)
        if torch.is_tensor(v):
            state[k] = v.cuda()


loss_fn = nn.BCELoss()


# 绘图函数
def gen_img_plot(model, epoch, test_input):
    # prediction = np.squeeze(model(test_input).detach().cpu().numpy()) # 去掉通道维度
    prediction = model(test_input).permute(0, 2, 3, 1).cpu().numpy()  # 将通道维度放在最后
    fig = plt.figure(figsize=(4, 4))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow((prediction[i] + 1) / 2)  # 将图像恢复到[0-1]之间，方便绘图
        plt.axis('off')
    plt.savefig('./data/GANimage_at_{}.png'.format(epoch))  # 把每一轮生成的图片保存到文件夹data中
    plt.show()


test_input = torch.randn(16, 100, device=device)  # 16个形状为100的随机数

# GAN的训练
D_loss = []
G_loss = []
# 循环训练
for epoch in range(200):
    d_epoch_loss = 0
    g_epoch_loss = 0
    count = len(train_loader)
    for step, (img, _) in enumerate(train_loader):
        img = img.to(device)
        size = img.size(0)
        random_noise = torch.randn(size, 100, device=device)  # 生成的noise做为Generator的输入
        d_optim.zero_grad()

        # 判别器损失
        real_output = dis(img)  # 对判别器输入真实的图片，real_output对真实图片的预测结果
        # 判别器对真实图片产生的损失 期望输出为全1，与全1的torch做loss
        d_real_loss = loss_fn(real_output, torch.ones_like(real_output))
        d_real_loss.backward()
        gen_img = gen(random_noise)  # 得到生成的图像
        # 判别器输入生成的图片，fake_output对生成图片的损失  detach截断梯度，梯度不在传递到generator中
        fake_output = dis(gen_img.detach())
        d_fake_loss = loss_fn(fake_output, torch.zeros_like(fake_output))  # 在生成图像上的损失
        d_fake_loss.backward()
        d_loss = d_real_loss + d_fake_loss  # 判别器包含两部分损失
        d_optim.step()

        # 生成器上的损失
        g_optim.zero_grad()
        fake_output = dis(gen_img)
        g_loss = loss_fn(fake_output,
                         torch.ones_like(fake_output))  # 希望生成器的图片被判别为1
        g_loss.backward()
        g_optim.step()

        with torch.no_grad():
            d_epoch_loss += d_loss
            g_epoch_loss += g_loss

    with torch.no_grad():
        d_epoch_loss /= count
        g_epoch_loss /= count
        D_loss.append(d_epoch_loss)
        G_loss.append(g_epoch_loss)
        print('epoch: %d\tg_loss = %.4f\t\td_loss = %.4f' % (epoch, g_epoch_loss, d_epoch_loss))
        gen_img_plot(gen, epoch, test_input)

    torch.save(gen.state_dict(), './model/gen.text')
    torch.save(dis.state_dict(), './model/dis.text')
    torch.save(g_optim.state_dict(), './model/g_optim.text')
    torch.save(d_optim.state_dict(), './model/d_optim.text')

