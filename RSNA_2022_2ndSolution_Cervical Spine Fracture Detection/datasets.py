from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

# 图像AUG策略
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate,
    CenterCrop, Resize, RandomCrop, GaussianBlur, JpegCompression, Downscale, ElasticTransform
)
import albumentations

from albumentations.pytorch import ToTensorV2


# 构造 dataset类
class TrainDataset(Dataset):
    def __init__(self, df, CFG, datadir, transform=None):
        self.df = df
        self.file_names = df['id'].values
        self.transform = transform
        self.CFG = CFG
        self.datadir = datadir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        image = np.load(f"{self.datadir}/seg_25d_image/{file_name}.npy") # 512 * 512 * 3
        mask = np.load(f"{self.datadir}/seg_25d_mask/{file_name}.npy") # 512 * 512 * 3

        # transform
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        image = image /255.0


        real_mask = np.zeros([self.CFG.num_classes, self.CFG.img_size, self.CFG.img_size])  # 8 * img_size * img_size
        for idx in range(self.CFG.num_classes):
            mask_bool = (mask[: ,: ,1] == (idx +1))
            real_mask[idx] = mask_bool

        image = np.transpose(image, (2, 0, 1)) # 3 * img_size * img_size
        mask = np.transpose(mask, (2, 0, 1)) # 3 * img_size * img_size

        return torch.from_numpy(image), torch.from_numpy(real_mask), torch.from_numpy(mask)

from utils.utils import *

class PesudoDataset(Dataset):
    def __init__(self, df, datadir, transform=None):
        self.df = df
        self.transform = transform
        self.datadir = datadir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        _, _, im2 = load_dicom(f"{self.datadir}/{row['path2']}")  # 512*512
        im2h = im2.shape[0]
        im2w = im2.shape[1]

        _, _, im1 = load_dicom(f"{self.datadir}/{row['path1']}") if row['path1'] != "nofile" else np.zeros(
            (im2h, im2w))  # 512*512
        _, _, im3 = load_dicom(f"{self.datadir}/{row['path3']}") if row['path3'] != "nofile" else np.zeros(
            (im2h, im2w))  # 512*512

        if im1.shape != (im2h, im2w):
            im1 = cv2.resize(im1, (im2w, im2h))
        if im3.shape != (im2h, im2w):
            im3 = cv2.resize(im3, (im2w, im2h))
        image_list = [im1, im2, im3]
        image = np.stack(image_list, axis=2)  # 512*512*3; 0-1

        # transform
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        # image = image/255.0
        image = np.transpose(image, (2, 0, 1))  # 3*img_size*img_size; 0-1
        return torch.from_numpy(image), row['StudyInstanceUID'], row['slice_num']


def get_transforms(data, CFG):
    if data == 'train':
        return Compose([
            Resize(CFG.img_size, CFG.img_size, interpolation=cv2.INTER_NEAREST),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
            OneOf([
                GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
                ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
            ], p=0.25),
            CoarseDropout(max_holes=8, max_height=CFG.img_size[0]//20, max_width=CFG.img_size[1]//20,
                            min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
            ], p=1.0)

    elif data == 'light_train':
        return Compose([
            Resize(CFG.img_size, CFG.img_size, interpolation=cv2.INTER_NEAREST),
            HorizontalFlip(p=0.5),
            # VerticalFlip(p=0.5),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
            OneOf([
                GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                # OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
                ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
            ], p=0.25),
            # CoarseDropout(max_holes=8, max_height=CFG.img_size[0]//20, max_width=CFG.img_size[1]//20,
            #              min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
            ], p=1.0)

    elif data == 'valid':
        return Compose([
            Resize(CFG.img_size, CFG.img_size, interpolation=cv2.INTER_NEAREST),
        ])

