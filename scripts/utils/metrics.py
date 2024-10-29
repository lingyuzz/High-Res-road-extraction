import torch

# 计算IoU
def calculate_iou(pred, target, smooth=1e-6):
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)

# 计算Dice系数
def dice_coefficient(pred, target, smooth=1e-6):
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

# 计算精确率 (Precision)
def precision(pred, target, smooth=1e-6):
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    true_positive = (pred_flat * target_flat).sum()
    predicted_positive = pred_flat.sum()
    return (true_positive + smooth) / (predicted_positive + smooth)

# 计算召回率 (Recall)
def recall(pred, target, smooth=1e-6):
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    true_positive = (pred_flat * target_flat).sum()
    actual_positive = target_flat.sum()
    return (true_positive + smooth) / (actual_positive + smooth)

# 计算F1分数
def f1_score(pred, target, smooth=1e-6):
    prec = precision(pred, target, smooth)
    rec = recall(pred, target, smooth)
    return 2 * (prec * rec) / (prec + rec + smooth)
