import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.dataset import RoadDataset  # 自定义的数据集类
from utils.model import ImprovedDeepLabV3Plus  # 模型定义
from utils.metrics import calculate_iou, dice_coefficient, precision, recall, f1_score  # 引入额外的评价指标

# 定义 mIoU 的计算函数
def calculate_miou(pred, mask):
    # True Positives, False Positives, True Negatives, False Negatives
    TP = ((pred == 1) & (mask == 1)).sum().item()
    FP = ((pred == 1) & (mask == 0)).sum().item()
    TN = ((pred == 0) & (mask == 0)).sum().item()
    FN = ((pred == 0) & (mask == 1)).sum().item()

    # IoU components
    iou_road = TP / (TP + FP + FN + 1e-6)  # 道路区域IoU
    iou_non_road = TN / (TN + FP + FN + 1e-6)  # 非道路区域IoU

    # 平均IoU（mIoU）
    miou = 0.5 * (iou_road + iou_non_road)
    return miou

# 定义评估函数
def evaluate_model(model, dataloader, device):
    model.eval()
    ious, dice_scores, precisions, recalls, f1_scores, mious = [], [], [], [], [], []

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            outputs = torch.sigmoid(outputs)  # 将输出转为0-1之间的概率值
            preds = (outputs > 0.5).float()  # 二值化预测结果

            # 计算每张图片的指标
            for pred, mask in zip(preds, masks):
                ious.append(calculate_iou(pred, mask))
                dice_scores.append(dice_coefficient(pred, mask))
                precisions.append(precision(pred, mask))
                recalls.append(recall(pred, mask))
                f1_scores.append(f1_score(pred, mask))

                # 计算并记录 mIoU
                mious.append(calculate_miou(pred, mask))

    # 计算各指标的平均值
    mean_iou = torch.mean(torch.tensor(ious)).item()
    mean_dice = torch.mean(torch.tensor(dice_scores)).item()
    mean_precision = torch.mean(torch.tensor(precisions)).item()
    mean_recall = torch.mean(torch.tensor(recalls)).item()
    mean_f1 = torch.mean(torch.tensor(f1_scores)).item()
    mean_miou = torch.mean(torch.tensor(mious)).item()

    # 打印结果
    print(f'平均IoU: {mean_iou:.4f}')
    print(f'平均Dice系数: {mean_dice:.4f}')
    print(f'平均精确率: {mean_precision:.4f}')
    print(f'平均召回率: {mean_recall:.4f}')
    print(f'平均F1分数: {mean_f1:.4f}')
    print(f'平均mIoU: {mean_miou:.4f}')

    return mean_iou, mean_dice, mean_precision, mean_recall, mean_f1, mean_miou

if __name__ == "__main__":

    # 模型和数据路径
    model_path = r"A:\High-Res road extraction\models\best_model.pth"
    test_images_path = r"A:\High-Res road extraction\data\data_match\train\img"
    test_masks_path = r"A:\High-Res road extraction\data\data_match\train\label"
    batch_size = 8

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据预处理和加载
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    # 加载测试数据集
    test_dataset = RoadDataset(test_images_path, test_masks_path, transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 加载模型
    model = ImprovedDeepLabV3Plus(num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=False)
    model.eval()

    # 评估模型
    evaluate_model(model, test_loader, device)
