https://zhuanlan.zhihu.com/p/617120030  Pytorch的详细使用

1、使用Sequential和OrderedDict来构建residual_inner_block,然后使用residual_inner_block构建residual_block，最后再Resnet(nn.Module)的init函数中构建整体框架， 需要给每一个层命名。

2、使用DataLoader加载数据集时使用drop_last



torchvision主要用来构建计算机视觉模型，它包含一些常用的数据集、模型、转换函数等等

Remark：png图像为4通道图像，除了三个颜色通道RGB以外还有一个透明度通道，需要用‘图像名.convert('RGB')转化成三通道图像'or'图像名.convert('L')'转化成单通道。



查看model.named_paremeters()等相关函数
