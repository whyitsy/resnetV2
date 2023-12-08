import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import json
import tqdm
# import torch_directml
# flower_photos directml 训练时1.5it/s 验证时6it/s batch_size=32
'''
FIFAR-10:
    directml:训练时1.5it/s 验证时12it/s batch_size=32 一个epoch 18min
'''
from Resnet34 import ResNet34

device = ("cuda" if torch.cuda.is_available() else "cpu")
# device = torch_directml.device()
# .to("xpu")还有bug
# device = ("xpu" if torch.xpu.is_available() else "cpu")
print(f"use {device}")

# 数据预处理
data_transform = {
    "train": transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
    ]),
    "test":transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
    ])
}

# 加载训练集
train_dataset = torchvision.datasets.CIFAR10(root="./data",transform=data_transform["train"],
                                             train=True,download=True)
train_len = len(train_dataset)

# 加载测试集
test_dataset = torchvision.datasets.CIFAR10(root="./data",transform=data_transform["test"],
                                            train=False,download=True)
test_len = len(test_dataset)

# 保存分类索引
# {'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}
# print(train_dataset.class_to_idx)  
'''
# class_dict = train_dataset.find_classes("./flower_photos/train")[1]
# class_index = [(value,key) for (key,value) in class_index]  
# class_index = [(value,key) for (key,value) in class_dict.items()]
class_dict是返回类别得到字典,但key是类别,value是index, 需要将k,v翻转一下
1、使用dict.items()返回字典的k,v对 2、使用python的一行复合代码将返回的k,v直接组成新的v,k对放入字典
'''
# class_dict = train_dataset.class_to_idx
# class_index = { value:key for key,value in class_dict.items()}
# with open("class_index.json","w") as f:
#     json.dump(class_index,f,indent=4)

# 创建网络 
resNet34 = ResNet34()

# 加载预训练权重----使用torch.load需要使用导入的Resnet类，返回网络的state_ditc
pre_weight_path = "./resnet34-b627a593.pth"
resNet34.load_state_dict(torch.load(pre_weight_path),strict=False)

# 修改分类个数
in_channel = resNet34.fc.in_features
resNet34.fc = torch.nn.Linear(in_channel,10)

# freeze后acc降低很多，还是要一起训练
# freeze除了最后一层的其它层权重----V1没有freeze
# for name,parm in resNet34.named_parameters():
#     # print(name) # 返回展平后每一层的名字
#     if "fc" not in name:
#         parm.requires_grad = False


# 超参数
batch_size = 32
epochs = 5 # 微调
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resNet34.parameters(),lr=0.0001)
save_path = "./resnet34.pth"

# 加载训练集迭代器----DataLoader()在哪个包里面？
# 这里有个bug？也许，在import的时候会有提示，但是在代码中就没有提示
# 但实际上好像是pylance的问题，卸载pylance后就有提示了
train_iter = DataLoader(dataset=train_dataset,batch_size=batch_size,
                                         shuffle=True,drop_last=True)
test_iter = DataLoader(dataset=test_dataset,batch_size=batch_size,
                                         shuffle=True,drop_last=True)

'''
问题1:optimizer.zero_grad()是每个epoch刷新还是每个batch刷新
    每个batch刷新,每个batch之后,optimizer会根据backward()计算的梯度更新权重,如果不刷新就会累计到下一个batch
问题2:resNet34.eval()和with torch.no_grad()区别
    with torch.no_grad()表示临时关闭梯度计算，从而提高评估过程的效率
    eval()是将整个模型设置为评估模式,所有的dropout和batch normalization层都不使用,也确保不计算梯度
    通常先调用 model.eval()，然后再使用 torch.no_grad() 临时关闭梯度计算
问题3:如何统计训练和测试时的数据
    loss_fn返回的loss是一个batch的总损失,所有的输出也是以batch为单位
    同时统计running_loss,最后输出一个epoch的running_loss/train_len
'''

resNet34 = resNet34.to(device)
loss_fn = loss_fn.to(device)

# 开始
for epoch in range(epochs):
    # 训练
    resNet34.train()
    running_loss = 0.0
    for step,data in tqdm.tqdm(enumerate(train_iter),desc="Train"):
        optimizer.zero_grad()
        imgs,labels = data
        outputs = resNet34(imgs.to(device))
        loss = loss_fn(outputs,labels.to(device)) # loss是这个batch的总损失
        running_loss += loss.item()
        loss.backward() # 反向传播计算梯度，这叫做自动求导吗？
        optimizer.step() # 更新权重

        # 每50个batch输出一次
        if (step+1) % 10 == 0:
            print("epoch{}_train,batch:{},loss:{:.5f}".format(epoch+1,step+1,loss))

    
    # 评估
    resNet34.eval()
    best_acc = 0.0 # 用于存储精度最高的模型
    with torch.no_grad():
        acc = 0.0
        total_right = 0
        for data in tqdm.tqdm(test_iter,desc="Validation"):
            imgs,labels = data
            outputs = resNet34(imgs.to(device))
            # 累计正确个数----刚刚没有把labels放入device,在GPU上跑的时候没有报错,会不会是把训练最后一个batch的labels拿来作对比了再跑一下就知道了
            total_right += (outputs.argmax(axis=1)==labels.to(device)).sum().item()
        
        acc = total_right/test_len
        if(best_acc < acc):
            best_acc = acc
            torch.save(resNet34.state_dict(),save_path)
        
        print("epoch{}_test,running_loss={:.5f},acc={:.5f}".format(epoch+1,running_loss/train_len,acc))


print("Training Finished!")

