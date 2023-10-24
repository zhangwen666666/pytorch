from torch.utils.tensorboard import SummaryWriter
import numpy
import cv2
writer = SummaryWriter("logs")  # 指定结果所储存的文件夹
image_path = "../../dataset/flower/train/daisy/5547758_eea9edfd54_n.jpg"
#PIL和numpy的形状是(H × W × C)，H、W、C范围在[0,255]
image = cv2.imread(image_path)  #读取图片  opencv读取的图片是numpy.ndarray格式
# print(type(image))
writer.add_image("test",image,2,dataformats="HWC")

# for i in range(100):
#     writer.add_scalar(tag="y = x", scalar_value=i, global_step=i)

writer.close()
