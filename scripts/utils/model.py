import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from efficientnet_pytorch import EfficientNet  # 导入EfficientNet

# CBAM 注意力机制
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, max(1, channels // reduction), 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(max(1, channels // reduction), channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力机制
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        out = self.sigmoid(out) * x

        # 空间注意力机制
        avg_out = torch.mean(out, dim=1, keepdim=True)
        max_out, _ = torch.max(out, dim=1, keepdim=True)
        out = self.spatial(torch.cat([avg_out, max_out], dim=1)) * out
        return out

# ASPP模块
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1),  # 1x1 卷积
            *(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate) for rate in atrous_rates)
        ])
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels * (len(atrous_rates) + 2))  # 确保它与ASPP输出通道一致
        self.relu = nn.ReLU()

    def forward(self, x):
        res = [conv(x) for conv in self.convs]
        global_avg = self.global_avg_pool(x)
        global_avg = F.interpolate(self.conv1x1(global_avg), size=x.shape[2:], mode='bilinear', align_corners=False)
        res.append(global_avg)
        res = torch.cat(res, dim=1)
        return self.relu(self.bn(res))

# DeepLabV3+ 改进版模型
class ImprovedDeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=1):
        super(ImprovedDeepLabV3Plus, self).__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0')

        # 1x1 卷积用于将通道数从 1280 减少到 256
        self.reduce_channels = nn.Conv2d(1280, 256, kernel_size=1)

        # ASPP 模块（输入通道数为 256）
        self.aspp = ASPP(256, 256, [12, 24, 36])

        # 添加一个卷积层，确保 ASPP 输出通道数保持 256
        self.aspp_out_conv = nn.Conv2d(1280, 256, kernel_size=1)

        # CBAM 模块，输入通道数为 256
        self.cbam = CBAM(256)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def extract_features(self, x):
        # 提取特征并调整通道数
        x = self.backbone.extract_features(x)
        #print(f"EfficientNet 输出的特征形状: {x.shape}")  # 调试信息

        # 使用 1x1 卷积将通道数从 1280 调整到 256
        x = self.reduce_channels(x)
        #print(f"Reduce Channels 调整后的特征形状: {x.shape}")  # 调试信息
        return x

    def forward(self, x):
        # 提取并调整后的特征传入 ASPP 和 CBAM
        x = self.extract_features(x)
        x = self.aspp(x)
        #print(f"ASPP 输出的特征形状: {x.shape}")  # 调试信息

        # 使用 1x1 卷积将 ASPP 输出通道数从 1280 调整到 256
        x = self.aspp_out_conv(x)
        #print(f"ASPP 输出通道数调整后的特征形状: {x.shape}")  # 调试信息

        x = self.cbam(x)
        #print(f"CBAM 输出的特征形状: {x.shape}")  # 调试信息

        x = self.classifier(x)
        
        # 对结果进行上采样，使其与输入的尺寸匹配
        x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)
        #print(f"上采样后的输出尺寸: {x.shape}")  # 最后调试输出的尺寸
        return x




# 自定义损失函数：结合BCE, Dice和Focal Loss
class BCEDiceFocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2):
        super(BCEDiceFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # BCE Loss
        bce = F.binary_cross_entropy_with_logits(inputs, targets)

        # Sigmoid激活
        inputs = torch.sigmoid(inputs)

        # Dice Loss
        smooth = 1e-6
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        # Focal Loss
        focal_loss = -self.alpha * (1 - inputs) ** self.gamma * targets * torch.log(inputs + smooth) - \
                     (1 - self.alpha) * inputs ** self.gamma * (1 - targets) * torch.log(1 - inputs + smooth)
        focal_loss = focal_loss.mean()

        return bce + dice_loss + focal_loss

# Cascade ASPP
class CascadeASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(CascadeASPP, self).__init__()
        self.aspp1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # 1x1卷积
        self.aspp2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=atrous_rates[0], padding=atrous_rates[0])
        self.aspp3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=atrous_rates[1], padding=atrous_rates[1])
        self.aspp4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=atrous_rates[2], padding=atrous_rates[2])
        self.conv = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)
        self.cbam = CBAM(out_channels)  # CBAM注意力机制

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.conv(x)
        x = self.cbam(x)  # 使用CBAM处理特征
        return x
