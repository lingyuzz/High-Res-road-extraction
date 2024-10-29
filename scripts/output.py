import torch
import matplotlib.pyplot as plt
import os
from utils.model import ImprovedDeepLabV3Plus  # 使用改进后的模型
from PIL import Image
from torchvision import transforms
import numpy as np

# 路径信息
image_dir = r"A:\High-Res road extraction\data\data_match\val\img"  # 验证图像路径
output_dir = r"A:\High-Res road extraction\results\visualization"
mask_output_dir = r"A:\High-Res road extraction\results\masks"  # 掩码输出路径
model_path = r"A:\High-Res road extraction\models\best_model.pth"

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = ImprovedDeepLabV3Plus(num_classes=1).to(device)
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)  # 安全加载
model.eval()

# 定义用于可视化的函数并保存结果
def visualize_and_save(image_path, model, device, output_path, mask_output_path, road_color=(1.0, 0.0, 0.0)):  # 归一化的红色
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    
    # 加载图像
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 模型预测
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        output = torch.sigmoid(output)
        prediction = (output > 0.5).float().cpu().squeeze().numpy()
    
    # 将原始图像转换为 numpy 数组并归一化
    image_np = np.array(image.resize((512, 512))) / 255.0

    # 道路区域的预测（掩码中的1表示道路）
    road_mask = prediction > 0.5
    
    # 将道路区域上色（设置为红色，或其他颜色，归一化为0-1）
    image_np[road_mask] = road_color  # 将道路区域标注为指定的颜色

    # 生成掩码图像并归一化
    mask_image = np.zeros_like(image_np[:, :, 0])  # 创建黑色背景
    mask_image[road_mask] = 1.0  # 将道路区域设置为白色（归一化）

    # 保存掩码图像
    mask_output_filename = os.path.join(mask_output_path, os.path.basename(image_path))
    Image.fromarray((mask_image * 255).astype(np.uint8)).save(mask_output_filename)  # 转换为0-255保存
    print(f"掩码图像已保存到: {mask_output_filename}")

    # 可视化输入图像、掩码图像和标注后的预测结果
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 输入图像
    axes[0].imshow(image_np)
    axes[0].set_title('Original Image')  # 修改标题为"原始图片"

    # 掩码图像
    axes[1].imshow(mask_image, cmap='gray')
    axes[1].set_title('Mask Image')  # 修改标题为"掩码图片"

    # 预测掩码的可视化
    axes[2].imshow(image_np)
    axes[2].set_title('Output Image')  # 修改标题为"输出图片"

    # 保存结果
    output_filename = os.path.join(output_path, os.path.basename(image_path).replace(".png", "_colored_result.png"))
    plt.savefig(output_filename)
    plt.close(fig)
    print(f"可视化结果已保存到: {output_filename}")

# 创建掩码输出目录
os.makedirs(mask_output_dir, exist_ok=True)

# 遍历所有图像
test_image_filenames = os.listdir(image_dir)
for image_filename in test_image_filenames:
    image_path = os.path.join(image_dir, image_filename)
    visualize_and_save(image_path, model, device, output_dir, mask_output_dir, road_color=(1.0, 0.0, 0.0))  # 归一化的红色标注道路
