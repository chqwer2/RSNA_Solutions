
# !pip install -q git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git

import sys

# Shut this TODO
# package_paths = [f'{libdir}pytorch-image-models-master']
# for pth in package_paths:
#     sys.path.append(pth)


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
from torch.utils.data import Dataset ,DataLoader
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
import gc




class CFG:
    seed = 42
    train_bs_ = 16  # train_batch_size
    valid_bs_ = 128  # valid_batch_size
    num_workers_ = 5

    device = 'GPU'  # ['TPU', 'GPU']
    nprocs = 1  # [1, 8]
    num_workers = num_workers_
    train_bs = train_bs_
    valid_bs = valid_bs_
    fold_num = 5

    target_cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "OT"]
    num_classes = 8

    normalize_mean = [0.4824, 0.4824, 0.4824]
    normalize_std = [0.22, 0.22, 0.22]

    fold_list = [0]

    model_arch = "efficientnet-b0"
    img_size = 512
    croped_img_size = 320  # 裁剪后的图片尺寸
