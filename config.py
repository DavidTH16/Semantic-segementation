import os
from dataclasses import dataclass

# Base Paths (Can be overridden for local runs)
BASE_DIR = '/kaggle/input/opencv-pytorch-segmentation-project-round2'
IMAGE_DIR = os.path.join(BASE_DIR, 'imgs/imgs')
MASK_DIR = os.path.join(BASE_DIR, 'masks/masks')
TRAIN_CSV = os.path.join(BASE_DIR, 'train.csv')
TEST_CSV = os.path.join(BASE_DIR, 'test.csv')

# Class names
CLASS_NAMES = ['background', 'person', 'bike', 'car',
               'drone', 'boat', 'animal', 'obstacle',
               'construction', 'vegetation', 'road', 'sky']

@dataclass
class SystemConf:
    cudnn_benchmark_enabled: bool = True
    cudnn_deterministic: bool = True
    epochs: int = 30
    learning_rate: float = 0.001
    batch_size: int = 12
    num_clases: int = 12
    decay_rate: float = 0.1
    device: str = 'cuda'
    num_workers: int = 4
    test_interval: int = 1

def get_mean():
    # values to be used in normalization based on ImageNet dataset
    # if you change the semantic model be sure the arch. model was trained
    # in this norm, mean values, if not change.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return mean, std
