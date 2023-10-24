from torch.utils.data import Dataset
from PIL import Image
import os


# print(help(Dataset))

class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir  # 根目录  "flower/train"
        self.label_dir = label_dir  # 标签    "daisy"
        self.path = os.path.join(root_dir, label_dir)  # 路径拼接  "flower/train/daisy"
        self.img_path = os.listdir(self.path)  # 将路径下的文件名组织成一个列表

    def __getitem__(self, idx):
        img_name = self.img_path[idx]  # 获取每一张图片的名字
        # 获取没一张图片的地址
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        # 读取图片
        img = Image.open(img_item_path)  # 图像
        label = self.label_dir  # 标签
        return img, label

    def __len__(self):
        return len(self.img_path)


root_dir = "../../dataset/flower/train"
daisy_label_dir = "daisy"
dandelion_label_dir = "dandelion"
roses_label_dir = "roses"
sunflowers_label_dir = "sunflowers"
tulips_label_dir = "tulips"

daisy_dataset = MyData(root_dir, daisy_label_dir)
dandelion_dataset = MyData(root_dir, dandelion_label_dir)
roses_dataset = MyData(root_dir, roses_label_dir)
sunflowers_dataset = MyData(root_dir, sunflowers_label_dir)
tulips_dataset = MyData(root_dir, tulips_label_dir)
train_dataset = daisy_dataset + dandelion_dataset + roses_dataset + sunflowers_dataset + tulips_dataset

print(len(train_dataset))