import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from utils.model import ImprovedDeepLabV3Plus, BCEDiceFocalLoss  # 导入改进的模型和损失函数
from albumentations import Compose, HorizontalFlip, RandomRotate90, RandomResizedCrop, Resize, Normalize  # 使用Albumentations进行数据增强
from albumentations.pytorch import ToTensorV2

# 数据路径配置
train_images = r"A:\High-Res road extraction\data\data_match\train\img"
train_masks = r"A:\High-Res road extraction\data\data_match\train\label"
val_images = r"A:\High-Res road extraction\data\data_match\val\img"

# 定义自定义数据集
class RoadExtractionDataset(Dataset):
    def __init__(self, images_dir, masks_dir=None, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.image_filenames = os.listdir(images_dir)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.image_filenames[idx])
        image = Image.open(image_path).convert('RGB')

        mask = None
        if self.masks_dir:
            mask_path = os.path.join(self.masks_dir, self.image_filenames[idx])
            mask = Image.open(mask_path).convert('L')

        # 仅在mask存在的情况下进行增强，否则只增强图像
        if self.transform:
            if mask is not None:
                augmented = self.transform(image=np.array(image), mask=np.array(mask))
                image = augmented['image']
                mask = augmented['mask']
            else:
                augmented = self.transform(image=np.array(image))
                image = augmented['image']

        # 如果存在mask，将其转换为二值化
        if mask is not None:
            mask = (mask > 0).float()

        return image, mask if mask is not None else image

# Albumentations数据增强
def get_transforms(phase):
    if phase == 'train':
        return Compose([
            RandomResizedCrop(512, 512, scale=(0.8, 1.0)),
            HorizontalFlip(p=0.5),
            RandomRotate90(p=0.5),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return Compose([
            Resize(512, 512),  # 确保验证集使用固定的输入尺寸
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

# 定义训练函数
def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs, device, accumulation_steps=4):
    best_iou = 0.0
    model_dir = r"A:\High-Res road extraction\models"  # 模型保存路径
    scaler = torch.amp.GradScaler('cuda')  # 使用新的GradScaler进行混合精度训练

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # 训练阶段
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()  # 清空梯度

        for i, (inputs, masks) in enumerate(train_loader):
            inputs = inputs.to(device)
            masks = masks.to(device)

            # 使用新的autocast进行混合精度训练
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                outputs = outputs.squeeze(1)  # 确保输出形状正确
                loss = criterion(outputs, masks) / accumulation_steps  # 累积梯度

            scaler.scale(loss).backward()

            # 每 accum_steps 次优化一次
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()  # 清空梯度，防止累积的梯度干扰下一次

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Training Loss: {epoch_loss:.4f}')

        scheduler.step()

        # 验证阶段跳过性能评估，仅可视化预测结果
        if val_loader:
            visualize_predictions(model, val_loader, device)

        # 保存最新模型
        torch.save(model.state_dict(), os.path.join(model_dir, 'latest_model.pth'))

    print('Training complete.')

# 可视化模型预测结果
def visualize_predictions(model, val_loader, device):
    model.eval()
    with torch.no_grad():
        for inputs in val_loader:
            # 如果 val_loader 只返回图像，那么 inputs 会是一个元组
            if isinstance(inputs, (list, tuple)):
                inputs = inputs[0]  # 获取第一个元素（即输入图像）
            
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze(1)
            outputs = torch.sigmoid(outputs)
            preds = (outputs > 0.5).float()

            # 这里可以添加代码将预测结果可视化，比如保存图片或显示
            # 例如：你可以使用 matplotlib 将预测结果和原始图像并排显示
            # 示例：plt.imshow(preds[0].cpu().numpy(), cmap='gray')

# 计算IoU的函数
def calculate_iou(pred, mask):
    pred_flat = pred.view(-1)
    mask_flat = mask.view(-1)
    intersection = (pred_flat * mask_flat).sum()
    union = pred_flat.sum() + mask_flat.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)


# 主函数
if __name__ == "__main__":
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据增强和预处理
    train_transforms = get_transforms('train')
    val_transforms = get_transforms('val')

    # 加载数据集
    train_dataset = RoadExtractionDataset(train_images, train_masks, transform=train_transforms)
    val_dataset = RoadExtractionDataset(val_images, transform=val_transforms)  # val没有masks

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)

    # 初始化模型、损失函数、优化器、学习率调度器
    model = ImprovedDeepLabV3Plus(num_classes=1).to(device)
    criterion = BCEDiceFocalLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # 训练模型
    num_epochs = 50  # 调整训练轮数
    train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs, device)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        