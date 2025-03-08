from PIL import Image
import torchvision
import torch.nn as nn
import torch

image_path = '/content/The-first-AI-project-to-study/x翼战斗机.png'
image = Image.open(image_path)
image = image.convert('RGB')
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor()
])
image = transform(image)

class HGL(nn.Module):
    def __init__(self):
        super(HGL, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2, stride=1),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, kernel_size=5, padding=2, stride=1),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2, stride=1),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=64 * 4 * 4, out_features=64),
            nn.Linear(in_features=64, out_features=10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
predictions = []

for i in range(10):
    # 加载模型时指定map_location='cpu'，确保权重在CPU上
    model = torch.load(f"hgl_{i}.pth", map_location='cpu', weights_only=False)
    image_tensor = torch.reshape(image, (1, 3, 32, 32))
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
    predicted_class = class_names[output.argmax(dim=1).item()]
    predictions.append(predicted_class)
    print(f"第{i+1}次预测结果：{predicted_class}")

# 统计最频繁出现的预测结果
from collections import Counter
most_common = Counter(predictions).most_common(1)
print("最终预测结果：", most_common[0][0])