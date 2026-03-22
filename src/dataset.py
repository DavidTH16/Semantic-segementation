import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class LandScape(Dataset):
    def __init__(self, images_dir, masks_dir, df, is_test=False, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.df = df
        self.is_test = is_test # False for Train/Val, True for Competition Test
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_id = self.df.iloc[index]['ImageID']
        
        # Load Image
        img_path = os.path.join(self.images_dir, f"{image_id}.jpg")
        image = np.array(Image.open(img_path))

        # Handle Test Mode (No Masks)
        if self.is_test:
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            return image, str(image_id)

        # Handle Train/Val Mode (With Masks)
        mask_path = os.path.join(self.masks_dir, f"{image_id}.png")
        mask = np.array(Image.open(mask_path))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']

        return image, mask
