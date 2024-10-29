import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class RoadDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.image_filenames = os.listdir(images_dir)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.masks_dir, self.image_filenames[idx])  # 假设mask与图像同名

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # 将mask加载为灰度图

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = (mask > 0).float()  # 将mask转换为二值图（0和1）

        return image, mask
