from PIL import Image
import torchvision
import torch.nn as nn
import torch
image_path = './x翼战斗机.png'
image=Image.open(image_path)
#print(image)
image=image.convert('RGB')
transform=torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                           torchvision.transforms.ToTensor()])
image=transform(image)
#print(image.shape)
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
for i in range(10):
    model=torch.load("hgl_{}.pth".format(i),weights_only=False)
    #print (model)
    image=torch.reshape(image,(1,3,32,32))
    model.eval()
    with torch.no_grad():
        output=model(image)
    #print(output)
    #print(output.argmax(dim=1))  # 输出概率最大的标签
    # 显示概率最大的类型的标签名
    class_names=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    print("第{}次预测结果：".format(i+1),class_names[output.argmax(dim=1).item()])
# 输出预测的最多的类型名称:
print("最终预测结果：",class_names[output.argmax(dim=1).item()])