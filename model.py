import torch.nn as nn
from torch import nn
import torch
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
if __name__=="__main__":
    hgl=HGL()
    input=torch.ones((64,3,32,32)) # 64 是 batch size, 3 是通道数, 32 是图片的高, 32 是图片的宽
    output=hgl(input)
    print(output.shape) # 输出的形状应该是 (64,10)