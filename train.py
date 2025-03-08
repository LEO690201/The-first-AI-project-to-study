import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
from model import *
import torch
from torch.utils.tensorboard import SummaryWriter
import time
train_data=torchvision.datasets.CIFAR10('./data',train=True,download=False,transform=torchvision.transforms.ToTensor())
test_data=torchvision.datasets.CIFAR10('./data',train=False,download=False,transform=torchvision.transforms.ToTensor())
train_data_size=len(train_data)  # 训练集大小
print("训练集大小：{}".format(train_data_size))
test_data_size=len(test_data)  # 测试集大小
print("测试集大小：{}".format(test_data_size))
# 利用dataloader加载数据
train_dataloader=DataLoader(train_data,batch_size=64)
test_dataloader=DataLoader(test_data,batch_size=64)
# 创建网络模型
class HGL(nn.Module):
    def __init__(self):
        super(HGL,self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(3,32,kernel_size=5,padding=2,stride=1),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32,32,kernel_size=5,padding=2,stride=1),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32,64,kernel_size=5,padding=2,stride=1),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=64*4*4,out_features=64),
            nn.Linear(in_features=64,out_features=10)
        )
    def forward(self,x):
        x=self.model(x)
        return x
hgl=HGL()
if torch.cuda.is_available():
    hgl=hgl.cuda()  # 利用GPU加速
# 损失函数
loss_fn=nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn=loss_fn.cuda()  # 利用GPU加速
# 优化器
learning_rate=1e-2
optimizer=torch.optim.SGD(hgl.parameters(),lr=learning_rate)
# 设置训练网络的参数
total_train_step=0  # 总训练步数
total_test_step=0   # 总测试步数
epoch=10            # 训练轮数
writer=SummaryWriter("./logs-model")
start_time=time.time()  # 记录训练开始时间
for i in range(epoch):
    print("-----------------第{}轮训练开始-----------------".format(i+1))
    # 训练步骤开始
    hgl.train()  # 开启训练模式
    for data in train_dataloader:
        imgs,targets=data 
        if torch.cuda.is_available():
            imgs=imgs.cuda()                           ###
            targets=targets.cuda()                     ###
        outputs=hgl(imgs)
        loss=loss_fn(outputs,targets)
        optimizer.zero_grad()   # 梯度清零
        loss.backward()  # 反向传播,用于计算参数的梯度
        optimizer.step()  # 使用优化器更新参数
        total_train_step+=1
        if total_train_step%100==0:  # 每100步打印一次损失
            end_time=time.time()
            print("训练步数：{}/{},耗时：{:.2f}s".format(total_train_step,train_data_size,(end_time-start_time)))
            print("训练步数：{}/{},损失：{:.4f}".format(total_train_step,train_data_size,loss))
            # loss_item()是指代当前张量的元素值，item()是指代当前张量的元素值，并将其转换为python的基本类型。
            writer.add_scalar('train_loss',loss.item(),total_train_step)
    # 测试步骤
    hgl.eval()  # 开启测试模式
    total_test_loss=0
    total_accuracy=0
    with torch.no_grad():  # 关闭梯度计算
        # 测试步骤开始
        for data in test_dataloader:
            imgs,targets=data
            if torch.cuda.is_available():
                imgs=imgs.cuda()                         ###
                targets=targets.cuda()                   ###
            outputs=hgl(imgs)
            loss=loss_fn(outputs,targets)
            total_test_loss+=loss.item()
            accuracy=(outputs.argmax(dim=1)==targets).sum()
            total_accuracy+=accuracy.item()
    print("整体测试集损失：{:.4f}".format(total_test_loss))
    print("整体测试集准确率：{:.4f}".format(total_accuracy))
    writer.add_scalar("test_loss",total_test_loss,total_test_step)
    total_test_step+=1
    torch.save(hgl,"hgl_{}.pth".format(i))
    print("模型已保存")
writer.close()
