              import torch
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from utils.model import ImprovedDeepLabV3Plus  # 引入改进后的模型定义
import argparse

# 定义函数：加载模型
def load_model(model_path):
    model = ImprovedDeepLabV3Plus(num_classes=1)  # 实例化模型
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # 加载模型权重
    model.eval()  # 切换到推理模式
    return model

# 定义函数：处理输入图像
def process_image(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # 根据模型输入尺寸调整
        transforms.ToTensor(),
    ])
    img = transform(img)
    img = img.unsqueeze(0)  # 增加batch维度
    return img

# 定义函数：保存预测结果
def save_output(pred, output_path):
    pred = torch.sigmoid(pred).squeeze().detach().numpy()  # 将Tensor转换为Numpy数组并使用sigmoid
    pred = (pred > 0.5).astype(np.uint8)  # 将预测结果二值化
    plt.imsave(output_path, pred, cmap='gray')  # 保存为灰度图

# 主函数：推理流程
def run_inference(model, input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在

    for image_name in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_name)
        output_path = os.path.join(output_dir, f"pred_{image_name}")

        print(f"Processing {image_path}...")

        # 加载并处理图像
        image = process_image(image_path)

        # 模型预测
        with torch.no_grad():
            output = model(image)

        # 保存预测结果
        save_output(output, output_path)

    print(f"推理完成，结果已保存至 {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="模型推理脚本")
    
    # 默认参数：当不提供命令行参数时，使用这些路径
    default_model_path = "A:\\High-Res road extraction\\models\\best_model.pth"
    default_input_dir = "A:\\High-Res road extraction\\data\\raw\\original_images\\test"
    default_output_dir = "A:\\High-Res road extraction\\results\\evaluation"
    
    # 添加参数解析
    parser.add_argument("--model", type=str, default=default_model_path, help="模型路径 (best_model.pth)")
    parser.add_argument("--input", type=str, default=default_input_dir, help="输入图像文件夹路径")
    parser.add_argument("--output", type=str, default=default_output_dir, help="输出结果文件夹路径")
    
    args = parser.parse_args()

    # 打印当前使用的参数路径
    print(f"使用模型: {args.model}")
    print(f"输入图像文件夹: {args.input}")
    print(f"输出结果文件夹: {args.output}")

    # 加载模型
    model = load_model(args.model)

    # 运行推理
    run_inference(model, args.input, args.output)
