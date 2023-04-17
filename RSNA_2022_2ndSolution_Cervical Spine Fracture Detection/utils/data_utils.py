import sys
import ast
from glob import glob
import cv2
from skimage import io
import os
from datetime import datetime
import time
import random
from tqdm import tqdm
from contextlib import contextmanager
import math

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score, log_loss
from sklearn import metrics
from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold
import torch
import torchvision
from torchvision import transforms
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from warmup_scheduler import GradualWarmupScheduler
import timm
import warnings
import joblib
from scipy.ndimage.interpolation import zoom
import nibabel as nib
import pydicom as dicom





def load_dicom(path):
    """
    This supports loading both regular and compressed JPEG images.
    See the first sell with `pip install` commands for the necessary dependencies
    """
    img = dicom.dcmread(path)
    img.PhotometricInterpretation = 'YBR_FULL'
    data = img.pixel_array
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return cv2.cvtColor(data, cv2.COLOR_GRAY2RGB), img


def process_seg(datadir, study_uid_list, dataframe_list):

    for file_name in tqdm(study_uid_list):
        ex_path = f"{datadir}/segmentations/{file_name}.nii"
        mask = nib.load(ex_path)
        mask = mask.get_fdata()  # convert to numpy array
        mask = mask[:, ::-1, ::-1].transpose(1, 0, 2)
        mask = np.clip(mask, 0, 8).astype(np.uint8)
        mask = np.ascontiguousarray(mask)

        train_image_path = glob(f"{datadir}/train_images/{file_name}/*")
        train_image_path = sorted(train_image_path, key=lambda x: int(x.split("/")[-1].replace(".dcm", "")))
        image_list = []
        for path in train_image_path:
            im, meta = load_dicom(path)
            image_list.append(im[:, :, 0])
        image = np.stack(image_list, axis=2)

        assert image.shape == mask.shape, f"Image and mask {file_name} should be the same size, but are {image.shape} and {mask.shape}"
        slice_num = image.shape[2]

        for i in range(1, slice_num - 1):
            image_25d = image[:, :, i - 1:i + 2]
            mask_25d = mask[:, :, i - 1:i + 2]
            assert image_25d.shape == mask_25d.shape == (512, 512,
                                                         3), f"Image and mask {file_name} should be (512, 512, 3), but are {image_25d.shape} and {mask_25d.shape}"
            image_save_path = f"{datadir}/seg_25d_image/{file_name}_{i}.npy"
            mask_save_path = f"{datadir}/seg_25d_mask/{file_name}_{i}.npy"
            np.save(image_save_path, image_25d)
            np.save(mask_save_path, mask_25d)
            dataframe_list.append([f"{file_name}_{i}", file_name, i, image_save_path, mask_save_path])
    return dataframe_list, study_uid_list

def set_folds(datadir, seg_25d_df):
    seg_25d_df["fold"] = -1
    gkf = GroupKFold(n_splits=5)
    for idx, (train_index, test_index) in enumerate(
            gkf.split(X=seg_25d_df, groups=seg_25d_df['StudyInstanceUID'].values)):
        seg_25d_df.loc[test_index, 'fold'] = idx

        for i in range(5):
            study_num = len(np.unique(seg_25d_df[seg_25d_df["fold"] == i]["StudyInstanceUID"]))
            print(f"fold{i} num: {study_num}")

        seg_25d_df.to_csv(f"{datadir}/seg_25d.csv", index=False)
