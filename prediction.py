import torch
import json
import PIL.Image
from torchvision import transforms

from Resnet34 import ResNet34

# 图像预处理
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 读取分类索引
class_inx = {}
with open("./class_index.json","r") as f:
    class_inx = json.load(f)

# 读取图片
photo_path = "flower_photos/prediction/向日葵2.jpg"
img = PIL.Image.open(photo_path)

img = data_transforms(img)
# 增加batch的维度 为1
img = torch.unsqueeze(img,0)

# 构建网络
resnet34 = ResNet34(5)
resnet34.load_state_dict(torch.load("resnet34.pth"))

resnet34.eval()
prediction = resnet34(img)
class_prediction = prediction.argmax(dim=1).item()

print("分类结果：",class_inx[str(class_prediction)])

