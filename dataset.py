# sace_cv/dataset.py
import os
import cv2
from torch.utils.data import Dataset

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')

class MoNuSeg(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(IMG_EXTS)])
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        name = self.images[index]
        img_path = os.path.join(self.image_dir, name)
        mask_path = os.path.join(self.mask_dir, name)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        # convert BGR->RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        mask = (mask >= 127).astype("float32")
        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask = aug["mask"]
        return image, mask
