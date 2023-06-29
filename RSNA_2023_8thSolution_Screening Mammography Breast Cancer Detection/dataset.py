import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, 
    CenterCrop, Resize, RandomCrop, GaussianBlur, JpegCompression, Downscale, ElasticTransform, Affine, ToFloat
)
import albumentations
from albumentations.pytorch import ToTensorV2


def get_transforms_8bit(data, img_size, normalize_mean, normalize_std):
    if data == 'train':
        return Compose([
            RandomResizedCrop(img_size[0], img_size[1], scale=(0.8, 1), ratio=(0.45, 0.55), p=1), 
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(rotate_limit=(-5, 5), p=0.3),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            JpegCompression(quality_lower=80, quality_upper=100, p=0.3),
            Affine(p=0.3),
            Normalize(mean=normalize_mean, std=normalize_std,),
            ToTensorV2(),
            ])
        
    elif data == 'valid':
        return Compose([
            Resize(img_size[0], img_size[1]),
            Normalize(mean=normalize_mean, std=normalize_std,),
            ToTensorV2(),
        ])

def get_transforms_16bit(data, img_size, normalize_mean, normalize_std):
    if data == 'train':
        return Compose([
            ToFloat(max_value=65535.0),
            RandomResizedCrop(img_size[0], img_size[1], scale=(0.8, 1), ratio=(0.45, 0.55), p=1), 
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(rotate_limit=(-5, 5), p=0.3),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            JpegCompression(quality_lower=80, quality_upper=100, p=0.3),
            Affine(p=0.3),
            ToTensorV2(),
            ])
        
    elif data == 'valid':
        return Compose([
            ToFloat(max_value=65535.0),
            Resize(img_size[0], img_size[1]),
            ToTensorV2(),
        ])
        
        
def get_transforms_16bit_light(data, img_size, normalize_mean, normalize_std):
    if data == 'train':
        return Compose([
            ToFloat(max_value=65535.0),
            RandomResizedCrop(img_size[0], img_size[1], scale=(0.9, 1), ratio=(0.45, 0.55), p=1), 
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.3),
            ShiftScaleRotate(rotate_limit=(-5, 5), p=0.3),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.3),
            JpegCompression(quality_lower=90, quality_upper=100, p=0.3),
            Affine(p=0.3),
            ToTensorV2(),
            ])
        
    elif data == 'valid':
        return Compose([
            ToFloat(max_value=65535.0),
            Resize(img_size[0], img_size[1]),
            ToTensorV2(),
        ])

def get_transforms_16bit_heavy(data, img_size, normalize_mean, normalize_std):
    if data == 'train':
        return Compose([
            ToFloat(max_value=65535.0),
            RandomResizedCrop(img_size[0], img_size[1], scale=(0.8, 1), ratio=(0.45, 0.55), p=1), 
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(rotate_limit=(-5, 5), p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.7),
            JpegCompression(quality_lower=80, quality_upper=100, p=0.5),
            Affine(p=0.7),
            ToTensorV2(),
            ])
        
    elif data == 'valid':
        return Compose([
            ToFloat(max_value=65535.0),
            Resize(img_size[0], img_size[1]),
            ToTensorV2(),
        ])
        
        
def get_transforms_16bit_heavy2(data, img_size, normalize_mean, normalize_std):
    if data == 'train':
        return Compose([
            ToFloat(max_value=65535.0),
            RandomResizedCrop(img_size[0], img_size[1], scale=(0.8, 1), ratio=(0.45, 0.55), p=1), 
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(rotate_limit=(-5, 5), p=0.8),
            RandomBrightnessContrast(brightness_limit=(-0.15,0.15), contrast_limit=(-0.15, 0.15), p=0.8),
            JpegCompression(quality_lower=80, quality_upper=100, p=0.8),
            Affine(p=0.8),
            ToTensorV2(),
            ])
        
    elif data == 'valid':
        return Compose([
            ToFloat(max_value=65535.0),
            Resize(img_size[0], img_size[1]),
            ToTensorV2(),
        ])


def mixup_augmentation(x:torch.Tensor, yc:torch.Tensor, alpha:float = 1.0):
    """
    Function which performs Mixup augmentation
    """
    assert alpha > 0, "Alpha must be greater than 0"
    assert x.shape[0] > 1, "Need more than 1 sample to apply mixup"

    lam = np.random.beta(alpha, alpha)
    rand_idx = torch.randperm(x.shape[0])
    
    mixed_x = lam * x + (1 - lam) * x[rand_idx, :]
    yc_j, yc_k = yc, yc[rand_idx]

    return mixed_x, yc_j, yc_k, lam



class BreastCancerDataSet_16bit(Dataset):
    def __init__(self, df, path, target_col, transforms=None):
        super().__init__()
        self.df = df
        self.path = path
        self.transforms = transforms
        self.target_col = target_col

    def __getitem__(self, i):
        path = f'{self.path}/{self.df.iloc[i].patient_id}/{self.df.iloc[i].image_id}.png'
        try:
            # 16 bit images
            img = Image.open(path)
            img = np.array(img).astype(np.uint16)
            if img.ndim == 2:
                img = np.repeat(img[:, :, np.newaxis], 3, axis=-1)
        except Exception as ex:
            print(path, ex)
            return None

        if self.transforms:
            img = self.transforms(image=np.array(img))["image"]

        if self.target_col in self.df.columns:
            cancer_target = torch.as_tensor(self.df.iloc[i].cancer)
            return img, cancer_target

        return img

    def __len__(self):
        return len(self.df)


class BreastCancerDataSet_8bit(Dataset):
    def __init__(self, df, path, target_col, transforms=None):
        super().__init__()
        self.df = df
        self.path = path
        self.transforms = transforms
        self.target_col = target_col

    def __getitem__(self, i):
        path = f'{self.path}/{self.df.iloc[i].patient_id}/{self.df.iloc[i].image_id}.png'
        try:
            img = Image.open(path).convert('RGB') # 8 bit images
        except Exception as ex:
            print(path, ex)
            return None

        if self.transforms:
            img = self.transforms(image=np.array(img))["image"]

        if self.target_col in self.df.columns:
            cancer_target = torch.as_tensor(self.df.iloc[i].cancer)
            return img, cancer_target

        return img

    def __len__(self):
        return len(self.df)
