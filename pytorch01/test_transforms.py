from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image

writer = SummaryWriter("logs")
img_path = "../../dataset/flower/train/daisy/5547758_eea9edfd54_n.jpg"
img = Image.open(img_path)
print(type(img))  # <class 'PIL.JpegImagePlugin.JpegImageFile'>
transform_tensor = transforms.ToTensor()  # 使用类实例化一个对象
tensor_img = transform_tensor(img)  # 调用__call__方法，需要一个参数(PIL或numpy格式的图片)
print(type(tensor_img))  # <class 'torch.Tensor'>
writer.add_image("tensor_img", tensor_img)

# 归一化
trans_norm = transforms.Normalize([50, 30, 10], [20, 60, 40])
img_norm = trans_norm(tensor_img)  # tensor_img是一个tensor类型的图片
writer.add_image("norm_img", img_norm)

# Resize
print(img.size)  # (320, 232)
trans_resize = transforms.Resize((512, 512))
resize_img = trans_resize(img)  # img是PIL格式图片
print(type(resize_img))  # <class 'PIL.Image.Image'>  输出也是PIL格式图片
print(resize_img.size)  # (512, 512)
resize_img_tensor = transform_tensor(resize_img)  # 转换为tensor格式
writer.add_image("re_img", resize_img_tensor)

# Compose
transform = transforms.Compose([
    transforms.Resize(512),  # 一个参数表示等比缩放
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

compose_img = transform(img)  # img是PIL格式图片
writer.add_image("compose_img", compose_img)

# RandomCrop()
trans_rc = transforms.RandomCrop(128)
transform = transforms.Compose([
    trans_rc,  # 随机裁剪
    transforms.ToTensor()
])
for i in range(10): #随机裁剪并转为tensor，循环10次
    img_rc = transform(img)
    writer.add_image("RandomCrop", img_rc, i)

writer.close()
