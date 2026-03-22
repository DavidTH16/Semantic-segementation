import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from .dataset import LandScape
import sys
import os

# To import from the root config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_mean

def get_data(image_path, mask_path, df, batch_size, num_workers, data_split='train'):
    # get mean & std
    mean, std = get_mean()
    width, height = (400, 400) # default 224*224 / 384 - 384
    
    if data_split == 'train':
        transform = A.Compose([
            A.SquareSymmetry(p=1.0),
            A.RandomBrightnessContrast(p=1),
            A.HueSaturationValue(p=1),
            # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.5, rotate_limit=15, p=1),
            A.Resize(width, height),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
        data_augm = LandScape(image_path, mask_path, df, is_test=False, transform=transform)
        shuffle = True
    elif data_split == 'val':
        transform = A.Compose([
            A.Resize(width, height),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
        data_augm = LandScape(image_path, mask_path, df, is_test=False, transform=transform)
        shuffle = False
    elif data_split == 'test':
        transform = A.Compose([
            A.Resize(width, height),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
        data_augm = LandScape(image_path, mask_path, df, is_test=True, transform=transform)
        shuffle = False
    else:
        raise ValueError("data_split must be one of 'train', 'val', or 'test'.")

    loader = DataLoader(
        dataset=data_augm,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True # speed up 
    )
    
    return loader
