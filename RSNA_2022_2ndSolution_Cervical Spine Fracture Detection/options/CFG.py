# !pip install -q git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git

import sys

# Shut this TODO
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
import math

import numpy as np
import pandas as pd
import sklearn
from sklearn import metrics
from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold
import torch
import torchvision
from torchvision import transforms
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.optim import Adam, SGD, AdamW

import timm
import warnings
import joblib
from scipy.ndimage.interpolation import zoom
import nibabel as nib

import gc


class CFG_stage1:

    train_bs_ = 16  # train_batch_size
    valid_bs_ = 32  # valid_batch_size
    num_workers_ = 5


    seed = 42
    device = 'GPU'
    nprocs = 1
    num_workers = num_workers_
    train_bs = train_bs_
    valid_bs = valid_bs_
    fold_num = 5

    target_cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "OT"]
    num_classes = 8

    accum_iter = 1
    max_grad_norm = 1000
    print_freq = 100
    normalize_mean = [0.4824, 0.4824, 0.4824]
    normalize_std = [0.22, 0.22, 0.22]

    suffix = "109"
    fold_list = [0]
    epochs = 15
    model_arch = "efficientnet-b0"
    img_size = 320
    optimizer = "AdamW"
    scheduler = "CosineAnnealingLR"
    loss_fn = "BCEWithLogitsLoss"
    scheduler_warmup = "GradualWarmupSchedulerV3"

    warmup_epo = 1
    warmup_factor = 10
    T_max = epochs - warmup_epo - 2 if scheduler_warmup == "GradualWarmupSchedulerV2" else \
        epochs - warmup_epo - 1 if scheduler_warmup == "GradualWarmupSchedulerV3" else epochs - 1

    lr = 5e-3
    min_lr = 1e-6  #
    weight_decay = 0

    n_early_stopping = 5


