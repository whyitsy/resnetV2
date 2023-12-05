from torch import nn
import torch.nn.functional as F

# bn的位置We adopt batch normalization (BN) right after each convolution and before activation
# 定义基础块
class BasicBlock(nn.Module):
    def __init__(self,in_channel,out_channel,use_1x1conv=False):
        super().__init__()
        # 使用1x1conv使得X和残差形状相同，可以相加
        if use_1x1conv:
            self.conv1 = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,
                               kernel_size=3,stride=2,padding=1,bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,
                                   kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(in_channels=out_channel,out_channels=out_channel,
                               kernel_size=3,stride=1,padding=1,bias=False)
        if use_1x1conv:
            self.conv3 =  nn.Conv2d(in_channels=in_channel,out_channels=out_channel,
                                    kernel_size=1,stride=2,bias=False)
        else:
            self.conv3 = None
        self.bn2 = nn.BatchNorm2d(num_features=out_channel)
    
    def forward(self,X):
        Y = self.conv1(X)
        Y = F.relu(self.bn1(Y))
        Y = self.conv2(Y)
        Y = self.bn2(Y)
        if(self.conv3 is not None):
           X = self.conv3(X)
        return F.relu(X+Y)


class ResNet34(nn.Module):
    # num_classes参数为了减少预测时要修改最后一层的操作
    def __init__(self,num_classes=1000):
        super().__init__()
        # 输入3,224,224
        self.conv0_1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,
                                 stride=2,padding=3,bias=False) # 64,112,112
        self.bn0_1 = nn.BatchNorm2d(num_features=64)
        # 这里需不需要relu()？
        self.relu0_1 = nn.ReLU()
        self.maxpool0_1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1) # 64,56,56
        # 使用_make_layer()获得后面的层
        self.layer1 = self._make_layer(in_channel=64,out_channel=64,num_basicBlock=3,is_first_block=True) # 64,56,56
        self.layer2 = self._make_layer(in_channel=64,out_channel=128,num_basicBlock=4) # 128,28,28
        self.layer3 = self._make_layer(in_channel=128,out_channel=256,num_basicBlock=6) # 256,14,14
        self.layer4 = self._make_layer(in_channel=256,out_channel=512,num_basicBlock=3) # 512,7,7
        # 定义最后的层
        self.globalavgpool = nn.AdaptiveAvgPool2d(1) # 512,1,1
        self.flatten = nn.Flatten() # 512
        self.fc = nn.Linear(512,num_classes) # num_classes=1000
        # 添加一个softmax
        self.softmax = nn.Softmax(dim=1)

    def forward(self,X):
        Y = self.conv0_1(X)
        Y = self.bn0_1(Y)
        Y = self.relu0_1(Y)
        Y = self.maxpool0_1(Y)
        Y = self.layer1(Y)
        Y = self.layer2(Y)
        Y = self.layer3(Y)
        Y = self.layer4(Y)
        Y = self.globalavgpool(Y)
        Y = self.flatten(Y)
        Y = self.fc(Y)
        Y = self.softmax(Y)
        return Y

    def _make_layer(self,in_channel,out_channel,num_basicBlock,is_first_block=False):
        # 准备list存储num_basicBlock个basicBlock
        layer = []
        for i in range(num_basicBlock):
            # 因为第一个block不进行大小和维度的变化，所以要单独处理
            if i == 0:
                if is_first_block:
                    layer.append(BasicBlock(in_channel=in_channel,out_channel=out_channel))
                else:
                    layer.append(BasicBlock(in_channel=in_channel,out_channel=out_channel,use_1x1conv=True))
            else:
                layer.append(BasicBlock(in_channel=out_channel,out_channel=out_channel))
        # return layer # 返回列表会导致Pytorch不会识别列表 网络就只有开头和结尾的层
        return nn.Sequential(*layer)
            

# 测试--使用children()函数获得可迭代的网络的各层(最外层)
# resnet34 = ResNet34()
# X = torch.rand((1,3,224,224))
# Y = resnet34(X)
# print(Y.shape)
# print(resnet34)
# for l in resnet34.children():
#     X = l(X)
#     print(l.__class__.__name__,'output shape:\t', X.shape)

