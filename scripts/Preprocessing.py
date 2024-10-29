import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from skimage.morphology import skeletonize
from shapely.geometry import LineString
import geopandas as gpd  # 导入用于保存 Shapefile 的库

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 定义模型结构
class DoubleConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Down(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.pool_conv = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.pool_conv(x)

class Up(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = torch.nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # 确保尺寸匹配
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = torch.nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class RoadSegmentationModel(torch.nn.Module):
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

# 路径配置
img_dir = r"A:\High-Res road extraction\data\data_match\train\img"
output_dir = r"A:\High-Res road extraction\data\output"
visualization_dir = r"A:\High-Res road extraction\data\visualization"

# 检查输出路径是否存在，不存在则创建
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(visualization_dir):
    os.makedirs(visualization_dir)

# 加载模型和参数
model = RoadSegmentationModel(n_channels=3, n_classes=1).to(device)
model.load_state_dict(torch.load(r"A:\High-Res road extraction\models\path_to_trained_model.pth", map_location=device))
model.eval()

# 定义图像预处理
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),  # 确保图像大小一致
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 遍历图像文件夹，处理每张图片
for img_name in tqdm(os.listdir(img_dir)):
    img_path = os.path.join(img_dir, img_name)
    output_path = os.path.join(output_dir, img_name)

    # 加载和预处理图像
    image = Image.open(img_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(device)  # 添加批次维度并加载到设备上

    # 使用模型预测
    with torch.no_grad():
        output = model(input_tensor)  # 输出为模型的预测结果
        output = torch.sigmoid(output)  # 使用Sigmoid函数将结果转为[0,1]区间
        output = (output > 0.5).float()  # 二值化，阈值为0.5

    # 将结果转换为numpy格式
    output_np = output.squeeze().cpu().numpy()  # 去掉批次维度并转为numpy数组

    # 去空洞和去碎片化
    kernel = np.ones((3, 3), np.uint8)
    closed_img = cv2.morphologyEx(output_np, cv2.MORPH_CLOSE, kernel)  # 填充空洞
    opened_img = cv2.morphologyEx(closed_img, cv2.MORPH_OPEN, kernel)  # 去除碎片化区域

    # 中心线矢量化
    skeleton = skeletonize(opened_img > 0)  # 骨架化生成中心线
    skeleton = skeleton.astype(np.uint8) * 255  # 将骨架化结果转换为二值图像格式

    # 保存骨架化图像
    skeleton_img_path = os.path.join(output_dir, f"skeleton_{img_name}")
    cv2.imwrite(skeleton_img_path, skeleton)

    # 生成二值化显示图像并保存到 visualization 目录
    visualization_image = np.where(opened_img > 1, 1, opened_img)  # 将大于1的值设置为1
    visualization_image = (visualization_image * 255).astype(np.uint8)  # 将1转换为255显示为白色
    visualization_img_path = os.path.join(visualization_dir, f"visualization_{img_name}")
    cv2.imwrite(visualization_img_path, visualization_image)

    # 矢量化中心线
    def skeleton_to_vector(skeleton):
        contours, _ = cv2.findContours(skeleton, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        vector_lines = []
        for contour in contours:
            if len(contour) > 1:  # 排除小噪声
                line = LineString(contour[:, 0, :])
                vector_lines.append(line)
        return vector_lines

    vector_lines = skeleton_to_vector(skeleton)

    # 道路宽度计算与缓冲
    road_polygons = []
    for line in vector_lines:
        width = 5  # 假设固定宽度，或使用自定义方法计算宽度
        buffered_line = line.buffer(width / 2)  # 使用宽度的一半生成缓冲区
        road_polygons.append(buffered_line)

    # 保存道路多边形矢量结果为 Shapefile，并指定坐标参考系统
    gdf = gpd.GeoDataFrame(geometry=road_polygons, crs="EPSG:4326")
    gdf.to_file(os.path.join(output_dir, f"road_{img_name}.shp"), driver="ESRI Shapefile")

print("道路栅格提取及矢量化处理完成，结果已保存至输出路径。")
