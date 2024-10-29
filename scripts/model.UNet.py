import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 定义模型
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.pool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # 确保尺寸匹配
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class RoadSegmentationModel(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(RoadSegmentationModel, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
        self.up1 = Up(1024 + 512, 512)
        self.up2 = Up(512 + 256, 256)
        self.up3 = Up(256 + 128, 128)
        self.up4 = Up(128 + 64, 64)
        
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# 数据集定义
class RoadDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform_img=None, transform_label=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform_img = transform_img
        self.transform_label = transform_label
        self.image_names = os.listdir(img_dir)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_names[idx])
        label_path = os.path.join(self.label_dir, self.image_names[idx])

        image = Image.open(img_path).convert("RGB")
        label = Image.open(label_path).convert("L")  # 单通道标签

        if self.transform_img:
            image = self.transform_img(image)
        if self.transform_label:
            label = self.transform_label(label)

        return image, label

# 训练参数
img_dir = r"A:\High-Res road extraction\data\data_match\train\img"
label_dir = r"A:\High-Res road extraction\data\data_match\train\label"
batch_size = 8
learning_rate = 1e-3
num_epochs = 10
save_path = r"A:\High-Res road extraction\models\path_to_trained_model.pth"

# 图像和标签的预处理
transform_img = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 标签仅需转换为张量
transform_label = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# 数据加载
train_dataset = RoadDataset(img_dir=img_dir, label_dir=label_dir, transform_img=transform_img, transform_label=transform_label)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 模型、损失函数和优化器
model = RoadSegmentationModel(n_channels=3, n_classes=1).to(device)  # 将模型加载到设备上
criterion = nn.BCEWithLogitsLoss()  # 使用二值交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练循环
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)  # 将数据加载到设备上

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# 保存模型权重
torch.save(model.state_dict(), save_path)
print(f"模型权重已保存至 {save_path}")
