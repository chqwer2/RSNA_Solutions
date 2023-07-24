
import os



os.environ["CUDA_VISIBLE_DEVICES"] = "0"
datadir = '../../rsna_cervical_spine'



libdir = '.'
outputdir = '.'
otherdir = '.'

train_bs_ = 16  # train_batch_size
valid_bs_ = 128  # valid_batch_size
num_workers_ = 2


# %% [markdown] {"id":"qu0pJE-UxU7u"}
# # CFG

# %% [code] {"executionInfo":{"elapsed":200843,"status":"ok","timestamp":1615451946014,"user":{"displayName":"徐铭","photoUrl":"https://lh3.googleusercontent.com/a-/AOh14GiblzBIU0N3bzEW-w5kOCgqAoahWcHJnTU5mI-A=s64","userId":"11772050295087096956"},"user_tz":-480},"id":"6fBgDehlNAAl","execution":{"iopub.status.busy":"2023-07-20T15:16:12.930819Z","iopub.execute_input":"2023-07-20T15:16:12.931485Z","iopub.status.idle":"2023-07-20T15:16:12.944109Z","shell.execute_reply.started":"2023-07-20T15:16:12.931449Z","shell.execute_reply":"2023-07-20T15:16:12.942623Z"}}
class CFG:
    seed = 42
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
    weight_path = f"./efficientnet-b0_109_fold0_epoch13_loss=0.052178958087850974.pth"





# %% [markdown] {"id":"p4w3IC-Qvcyq"}
# # Import

# %% [code] {"executionInfo":{"elapsed":222046,"status":"ok","timestamp":1615451967224,"user":{"displayName":"徐铭","photoUrl":"https://lh3.googleusercontent.com/a-/AOh14GiblzBIU0N3bzEW-w5kOCgqAoahWcHJnTU5mI-A=s64","userId":"11772050295087096956"},"user_tz":-480},"id":"4U1yFOTSM5Jw","outputId":"5e7b5f5c-88ed-40ff-f025-4c28bff1fd43","execution":{"iopub.status.busy":"2023-07-20T15:16:12.948741Z","iopub.execute_input":"2023-07-20T15:16:12.949827Z","iopub.status.idle":"2023-07-20T15:16:19.393643Z","shell.execute_reply.started":"2023-07-20T15:16:12.949780Z","shell.execute_reply":"2023-07-20T15:16:19.392329Z"}}
import sys;

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
from torch.utils.data import Dataset, DataLoader
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


from torch.cuda.amp import autocast, GradScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# %% [code] {"executionInfo":{"elapsed":225887,"status":"ok","timestamp":1615451971076,"user":{"displayName":"徐铭","photoUrl":"https://lh3.googleusercontent.com/a-/AOh14GiblzBIU0N3bzEW-w5kOCgqAoahWcHJnTU5mI-A=s64","userId":"11772050295087096956"},"user_tz":-480},"id":"8X-g-a_hM_zN","execution":{"iopub.status.busy":"2023-07-20T15:16:19.395656Z","iopub.execute_input":"2023-07-20T15:16:19.396129Z","iopub.status.idle":"2023-07-20T15:16:19.421277Z","shell.execute_reply.started":"2023-07-20T15:16:19.396090Z","shell.execute_reply":"2023-07-20T15:16:19.420078Z"}}
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(CFG.seed)


def get_score(y_true, y_pred):
    scores = []
    for i in range(y_true.shape[1]):
        score = roc_auc_score(y_true[:, i], y_pred[:, i])
        scores.append(score)
    avg_score = np.mean(scores)
    return avg_score, scores


# 日志记录函数
def init_logger(log_file=outputdir + '/train.log'):
    from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def get_timediff(time1, time2):
    minute_, second_ = divmod(time2 - time1, 60)
    return f"{int(minute_):02d}:{int(second_):02d}"


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def get_img(path):
    im_bgr = cv2.imread(path)
    im_rgb = im_bgr[:, :, ::-1]
    return im_rgb


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
    # data = (data * 255).astype(np.uint8)
    return data




# %% [code] {"execution":{"iopub.status.busy":"2023-07-20T15:16:19.491952Z","iopub.execute_input":"2023-07-20T15:16:19.492814Z","iopub.status.idle":"2023-07-20T15:16:19.538703Z","shell.execute_reply.started":"2023-07-20T15:16:19.492776Z","shell.execute_reply":"2023-07-20T15:16:19.537539Z"}}
study_train_df = pd.read_csv(f'{datadir}/train.csv')
print('train_df shape:', study_train_df.shape)


# %% [code] {"execution":{"iopub.status.busy":"2023-07-20T15:16:19.540380Z","iopub.execute_input":"2023-07-20T15:16:19.541379Z","iopub.status.idle":"2023-07-20T15:16:19.599408Z","shell.execute_reply.started":"2023-07-20T15:16:19.541339Z","shell.execute_reply":"2023-07-20T15:16:19.598447Z"}}
seg_paths = glob(f"{datadir}/segmentations/*")
seg_gt_list = [path.split('/')[-1][:-4] for path in seg_paths]
print(len(seg_gt_list))


# %% [code] {"execution":{"iopub.status.busy":"2023-07-20T15:16:19.601025Z","iopub.execute_input":"2023-07-20T15:16:19.601734Z","iopub.status.idle":"2023-07-20T15:16:19.615692Z","shell.execute_reply.started":"2023-07-20T15:16:19.601693Z","shell.execute_reply":"2023-07-20T15:16:19.614176Z"}}
study_train_df = study_train_df[~study_train_df["StudyInstanceUID"].isin(seg_gt_list)]


# %% [code] {"execution":{"iopub.status.busy":"2023-07-20T15:16:19.617521Z","iopub.execute_input":"2023-07-20T15:16:19.618822Z","iopub.status.idle":"2023-07-20T15:16:24.411722Z","shell.execute_reply.started":"2023-07-20T15:16:19.618781Z","shell.execute_reply":"2023-07-20T15:16:24.410675Z"}}
train_slice_list = []
# for file_name in tqdm(study_train_df["StudyInstanceUID"].values):
for file_name in tqdm(study_train_df["StudyInstanceUID"].values[:30]):

    train_image_path = glob(f"{datadir}/train_images/{file_name}/*")
    train_image_path = sorted(train_image_path, key=lambda x: int(x.split("/")[-1].replace(".dcm", "")))
    for path_idx in range(len(train_image_path)):
        path1 = "nofile" if path_idx - 1 < 0 else train_image_path[path_idx - 1].replace(f"{datadir}/", "")
        path2 = train_image_path[path_idx].replace(f"{datadir}/", "")
        path3 = "nofile" if path_idx + 1 >= len(train_image_path) else train_image_path[path_idx + 1].replace(
            f"{datadir}/", "")
        slice_num = int(path2.split("/")[-1].replace(".dcm", ""))
        train_slice_list.append([f"{file_name}_{slice_num}", file_name, slice_num, path1, path2, path3])

# %% [code] {"execution":{"iopub.status.busy":"2023-07-20T15:16:24.413118Z","iopub.execute_input":"2023-07-20T15:16:24.413495Z","iopub.status.idle":"2023-07-20T15:16:24.582459Z","shell.execute_reply.started":"2023-07-20T15:16:24.413458Z","shell.execute_reply":"2023-07-20T15:16:24.581191Z"}}
train_df = pd.DataFrame(train_slice_list, columns=["id", "StudyInstanceUID", "slice_num", "path1", "path2", "path3"])
train_df = train_df.sort_values(['StudyInstanceUID', 'slice_num'], ascending=[True, True]).reset_index(drop=True)
train_df.to_csv(f'train_slice_list.csv', index=False)




# %% [markdown] {"id":"RXBaZcUpQLqw"}
# # Dataset

# %% [code] {"executionInfo":{"elapsed":225885,"status":"ok","timestamp":1615451971076,"user":{"displayName":"徐铭","photoUrl":"https://lh3.googleusercontent.com/a-/AOh14GiblzBIU0N3bzEW-w5kOCgqAoahWcHJnTU5mI-A=s64","userId":"11772050295087096956"},"user_tz":-480},"id":"fuxNse_1M_r0","execution":{"iopub.status.busy":"2023-07-20T15:16:24.584346Z","iopub.execute_input":"2023-07-20T15:16:24.584783Z","iopub.status.idle":"2023-07-20T15:16:24.599437Z","shell.execute_reply.started":"2023-07-20T15:16:24.584744Z","shell.execute_reply":"2023-07-20T15:16:24.598194Z"}}
# 构造 dataset类
class TrainDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        im2 = load_dicom(f"{datadir}/{row['path2']}")  # 512*512
        im2h = im2.shape[0]
        im2w = im2.shape[1]

        im1 = load_dicom(f"{datadir}/{row['path1']}") if row['path1'] != "nofile" else np.zeros(
            (im2h, im2w))  # 512*512
        im3 = load_dicom(f"{datadir}/{row['path3']}") if row['path3'] != "nofile" else np.zeros(
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

    # %% [code] {"executionInfo":{"elapsed":225884,"status":"ok","timestamp":1615451971077,"user":{"displayName":"徐铭","photoUrl":"https://lh3.googleusercontent.com/a-/AOh14GiblzBIU0N3bzEW-w5kOCgqAoahWcHJnTU5mI-A=s64","userId":"11772050295087096956"},"user_tz":-480},"id":"kGjOdUreM_jF","execution":{"iopub.status.busy":"2023-07-20T15:16:24.601306Z","iopub.execute_input":"2023-07-20T15:16:24.602040Z","iopub.status.idle":"2023-07-20T15:16:25.655833Z","shell.execute_reply.started":"2023-07-20T15:16:24.601999Z","shell.execute_reply":"2023-07-20T15:16:25.654534Z"}}


# 图像AUG策略
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout,
    ShiftScaleRotate,
    CenterCrop, Resize, RandomCrop, GaussianBlur, JpegCompression, Downscale, ElasticTransform
)
import albumentations

from albumentations.pytorch import ToTensorV2


def get_transforms(data):
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
            CoarseDropout(max_holes=8, max_height=CFG.img_size[0] // 20, max_width=CFG.img_size[1] // 20,
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


# %% [markdown] {"id":"GbWy6sEbRl8P"}
# # Model

# %% [code] {"executionInfo":{"elapsed":229120,"status":"ok","timestamp":1615451974323,"user":{"displayName":"徐铭","photoUrl":"https://lh3.googleusercontent.com/a-/AOh14GiblzBIU0N3bzEW-w5kOCgqAoahWcHJnTU5mI-A=s64","userId":"11772050295087096956"},"user_tz":-480},"id":"ty22GW9qRkDk","execution":{"iopub.status.busy":"2023-07-20T15:16:25.657979Z","iopub.execute_input":"2023-07-20T15:16:25.658456Z","iopub.status.idle":"2023-07-20T15:16:26.936387Z","shell.execute_reply.started":"2023-07-20T15:16:25.658415Z","shell.execute_reply":"2023-07-20T15:16:26.935157Z"}}
import segmentation_models_pytorch as smp


def build_model():
    model = smp.Unet(
        encoder_name=CFG.model_arch,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=CFG.num_classes,  # model output channels (number of classes in your dataset)
        activation=None,
    )
    model.to(device)
    return model


def load_model(path):
    model = build_model()
    model.load_state_dict(torch.load(path)["model"])
    model.eval()
    return model


# %% [markdown] {"id":"MQhPYPmZe2gC"}
# # Inference

# %% [code] {"execution":{"iopub.status.busy":"2023-07-20T15:16:26.938159Z","iopub.execute_input":"2023-07-20T15:16:26.938794Z","iopub.status.idle":"2023-07-20T15:16:26.946462Z","shell.execute_reply.started":"2023-07-20T15:16:26.938748Z","shell.execute_reply":"2023-07-20T15:16:26.945209Z"}}
# os.makedirs(f"{outputdir}/train_voxel", exist_ok=True)
# os.makedirs(f"{outputdir}/train_voxel_mask", exist_ok=True)
# for filename in train_df["StudyInstanceUID"].values:
#     os.makedirs(f"{outputdir}/train_mask/{filename}", exist_ok=True)

# %% [code] {"execution":{"iopub.status.busy":"2023-07-20T15:16:26.948798Z","iopub.execute_input":"2023-07-20T15:16:26.949667Z","iopub.status.idle":"2023-07-20T15:16:26.972843Z","shell.execute_reply.started":"2023-07-20T15:16:26.949625Z","shell.execute_reply":"2023-07-20T15:16:26.971472Z"}}
slice_class_list = []
voxel_crop_list = []


def crop_voxel(voxel_mask, last_f_name):
    area_thr = 10
    # x
    x_list = []
    length = voxel_mask.shape[0]
    for i in range(length):
        if torch.count_nonzero(voxel_mask[i]).item() >= area_thr:
            x_list.append(i)
            break
    else:
        x_list.append(0)

    for i in range(length - 1, -1, -1):
        if torch.count_nonzero(voxel_mask[i]).item() >= area_thr:
            x_list.append(i)
            break
    else:
        x_list.append(length - 1)

    # y
    y_list = []
    length = voxel_mask.shape[1]
    for i in range(length):
        if torch.count_nonzero(voxel_mask[:, i]).item() >= area_thr:
            y_list.append(i)
            break
    else:
        y_list.append(0)

    for i in range(length - 1, -1, -1):
        if torch.count_nonzero(voxel_mask[:, i]).item() >= area_thr:
            y_list.append(i)
            break
    else:
        y_list.append(length - 1)

    # z
    z_list = []
    length = voxel_mask.shape[2]
    for i in range(length):
        if torch.count_nonzero(voxel_mask[:, :, i]).item() >= area_thr:
            z_list.append(i)
            break
    else:
        z_list.append(0)

    for i in range(length - 1, -1, -1):
        if torch.count_nonzero(voxel_mask[:, :, i]).item() >= area_thr:
            z_list.append(i)
            break
    else:
        z_list.append(length - 1)
    # croped_voxel = voxels[x_list[0]:x_list[1]+1, y_list[0]:y_list[1]+1, z_list[0]:z_list[1]+1]
    try:
        croped_voxel_mask = voxel_mask[x_list[0]:x_list[1] + 1, y_list[0]:y_list[1] + 1, z_list[0]:z_list[1] + 1]
    except:
        print(
            f"last_f_name:{last_f_name}, voxel_mask.shape:{voxel_mask.shape}, x_list:{x_list}, y_list:{y_list}, z_list:{z_list}")
        x_list = [0, voxel_mask.shape[0] - 1];
        y_list = [0, voxel_mask.shape[1] - 1];
        z_list = [0, voxel_mask.shape[2] - 1]
        croped_voxel_mask = voxel_mask
    voxel_crop_list.append(
        [last_f_name, voxel_mask.shape[1], x_list[0], x_list[1] + 1, y_list[0], y_list[1] + 1, z_list[0],
         z_list[1] + 1])

    # croped_voxel = croped_voxel.to('cpu').numpy() # bs*img_size*img_size; 0-8 classes
    croped_voxel_mask = croped_voxel_mask.to('cpu').numpy().astype(np.uint8)  # bs*img_size*img_size; 0-8 classes
    for x_idx in range(croped_voxel_mask.shape[0]):
        slice_mask = croped_voxel_mask[x_idx]

        unique, counts = np.unique(slice_mask, return_counts=True)
        if len(unique) == 1 and unique[0] == 0:
            slice_class_list.append([last_f_name, x_idx, x_idx + x_list[0], 0])
        elif unique[0] == 0:
            unique = unique[1:]
            counts = counts[1:]
            slice_class_list.append([last_f_name, x_idx, x_idx + x_list[0] + 1, unique[counts.argmax()]])
        else:
            slice_class_list.append([last_f_name, x_idx, x_idx + x_list[0] + 1, unique[counts.argmax()]])

    return None, croped_voxel_mask


# %% [code] {"execution":{"iopub.status.busy":"2023-07-20T15:16:26.974367Z","iopub.execute_input":"2023-07-20T15:16:26.975749Z","iopub.status.idle":"2023-07-20T15:16:27.255617Z","shell.execute_reply.started":"2023-07-20T15:16:26.975708Z","shell.execute_reply":"2023-07-20T15:16:27.254115Z"}}
torch.cuda.empty_cache()
import gc

gc.collect()

# %% [code] {"execution":{"iopub.status.busy":"2023-07-20T15:16:27.257786Z","iopub.execute_input":"2023-07-20T15:16:27.258428Z","iopub.status.idle":"2023-07-20T15:16:31.889689Z","shell.execute_reply.started":"2023-07-20T15:16:27.258387Z","shell.execute_reply":"2023-07-20T15:16:31.888586Z"}}
model = load_model(CFG.weight_path)
model.eval()



test_dataset = TrainDataset(train_df, transform=get_transforms("valid")) # get_transforms("valid")
test_loader = DataLoader(test_dataset, batch_size=CFG.valid_bs, shuffle=False, num_workers=CFG.num_workers, pin_memory=True, drop_last=False)

model = load_model(CFG.weight_path)
model.eval()
last_f_name = ""
voxel_mask = []
# voxels = []
for step, (images, file_names, n_slice) in tqdm(enumerate(test_loader),total=len(test_loader)):
    images = images.to(device, dtype=torch.float) # bs*3*image_size*image_size
    batch_size = images.size(0)
    with torch.no_grad():
        y_pred = model(images) # [B, 8, H, W]
    y_pred = y_pred.sigmoid()
    slice_mask_max = torch.max(y_pred, 1) # bs*img_size*img_size
    slice_mask = torch.where((slice_mask_max.values)>0.5, slice_mask_max.indices+1, 0) # bs*img_size*img_size; 0-8 classes
    slice_mask = torch.where(slice_mask==8,0,slice_mask).type(torch.uint8)
    # slice_mask = slice_mask.to('cpu').numpy().astype(np.uint8) # bs*img_size*img_size; 0-8 classes
    # slice_image = images[:, 1, :, :] # bs*img_size*img_size

    start_idx = 0
    for bs_idx in range(batch_size):
        f_name = file_names[bs_idx]
        if f_name != last_f_name:
            voxel_mask.append(slice_mask[start_idx:bs_idx])
            # voxels.append(slice_image[start_idx:bs_idx])
            voxel_mask = torch.cat(voxel_mask, dim=0) # n_slice*img_size*img_size; 0-8 classes
            # voxels = torch.cat(voxels, dim=0) # n_slice*img_size*img_size
            if len(voxel_mask) > 0:
                croped_voxel, croped_voxel_mask = crop_voxel(voxel_mask, last_f_name)
            last_f_name = f_name
            start_idx = bs_idx
            voxel_mask = []
            # voxels = []
        elif bs_idx == batch_size-1:
            voxel_mask.append(slice_mask[start_idx:batch_size])
            # voxels.append(slice_image[start_idx:batch_size])
voxel_mask = torch.cat(voxel_mask, dim=0)
if len(voxel_mask) > 0:
    croped_voxel, croped_voxel_mask = crop_voxel(voxel_mask, last_f_name)



# %% [markdown]
# ### replace 87 GT

# %% [code] {"execution":{"iopub.status.busy":"2023-07-20T15:17:00.024741Z","iopub.status.idle":"2023-07-20T15:17:00.025444Z","shell.execute_reply.started":"2023-07-20T15:17:00.025100Z","shell.execute_reply":"2023-07-20T15:17:00.025131Z"}}
# slice_class_list = []
# voxel_crop_list = []
for file_name in tqdm(seg_gt_list):
    ex_path = f"{datadir}/segmentations/{file_name}.nii"
    mask = nib.load(ex_path)
    mask = mask.get_fdata()  # convert to numpy array
    mask = mask[:, ::-1, ::-1].transpose(1, 0, 2)
    mask = np.clip(mask, 0, 8).astype(np.uint8)
    mask = np.where(mask == 8, 0, mask)
    mask = np.ascontiguousarray(mask)  # 512*512*slice
    # print("mask:", mask.shape, mask.max())  # (512, 512, 195) 7

    if mask.shape[0] != 512 or mask.shape[1] != 512:
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
    mask = mask.transpose(2, 0, 1)
    assert mask.shape[1] == mask.shape[2] == 512
    mask = torch.from_numpy(mask).to(device, dtype=torch.uint8)
    croped_voxel, croped_voxel_mask = crop_voxel(mask, file_name)



# %% [markdown]
# # post-progress

# %% [code] {"execution":{"iopub.status.busy":"2023-07-20T15:17:00.032717Z","iopub.status.idle":"2023-07-20T15:17:00.033969Z","shell.execute_reply.started":"2023-07-20T15:17:00.033682Z","shell.execute_reply":"2023-07-20T15:17:00.033712Z"}}
import pandas as pd
import numpy as np
from tqdm import tqdm

# datadir = '../kingston'

# %% [markdown]
# ### voxel_crop_list & vertebra_class

# %% [code] {"execution":{"iopub.status.busy":"2023-07-20T15:17:00.035461Z","iopub.status.idle":"2023-07-20T15:17:00.036459Z","shell.execute_reply.started":"2023-07-20T15:17:00.036184Z","shell.execute_reply":"2023-07-20T15:17:00.036211Z"}}
# voxel_crop_df = pd.read_csv(f"{datadir}/voxel_crop.csv")
voxel_crop_df = pd.DataFrame(voxel_crop_list,
                             columns=["StudyInstanceUID", "before_image_size", "x0", "x1", "y0", "y1", "z0",
                                      "z1"]).sort_values(by=["StudyInstanceUID"])
voxel_crop_df.to_csv(f"{datadir}/voxel_crop.csv", index=False)
voxel_crop_df  # 每个study的整体crop坐标



# %% [code] {"execution":{"iopub.status.busy":"2023-07-20T15:17:00.038834Z","iopub.status.idle":"2023-07-20T15:17:00.039669Z","shell.execute_reply.started":"2023-07-20T15:17:00.039355Z","shell.execute_reply":"2023-07-20T15:17:00.039382Z"}}
# slice_class_df = pd.read_csv(f"{datadir}/slice_class.csv")
slice_class_df = pd.DataFrame(slice_class_list, columns=["StudyInstanceUID", "new_slice_num", "old_slice_num",
                                                         "vertebra_class"]).sort_values(
    by=["StudyInstanceUID", "new_slice_num"])
slice_class_df.to_csv(f"{datadir}/slice_class.csv", index=False)
slice_class_df  # 每张slice的所属vertebra_class(preds)



# %% [code] {"execution":{"iopub.status.busy":"2023-07-20T15:17:00.042239Z","iopub.status.idle":"2023-07-20T15:17:00.049127Z","shell.execute_reply.started":"2023-07-20T15:17:00.048822Z","shell.execute_reply":"2023-07-20T15:17:00.048852Z"}}
study_id_list = []
slice_num_list = []
for file_name in tqdm(voxel_crop_df["StudyInstanceUID"].values, total=len(voxel_crop_df)):
    train_image_path = glob(f"{datadir}/train_images/{file_name}/*")
    train_image_path = sorted(train_image_path, key=lambda x: int(x.split("/")[-1].replace(".dcm", "")))
    slice_cnt = len(train_image_path)

    study_id_list.extend([file_name] * slice_cnt)
    slice_num_list.extend([int(x.split("/")[-1].replace(".dcm", "")) for x in train_image_path])

all_slice_df = pd.DataFrame({"StudyInstanceUID": study_id_list, "slice_num": slice_num_list})
all_slice_df.to_csv(f"{datadir}/all_slice_df.csv", index=False)
# all_slice_df = pd.read_csv(f"{datadir}/all_slice_df.csv")
print(all_slice_df.shape)
all_slice_df.head(3)



# %% [markdown]
# ## gen new_df

# %% [code] {"execution":{"iopub.status.busy":"2023-07-20T15:17:00.052186Z","iopub.status.idle":"2023-07-20T15:17:00.055815Z","shell.execute_reply.started":"2023-07-20T15:17:00.055452Z","shell.execute_reply":"2023-07-20T15:17:00.055482Z"}}
new_df = []
for idx, study_id, _, x0, x1, _, _, _, _, in tqdm(voxel_crop_df.itertuples(), total=len(voxel_crop_df)):
    one_study = all_slice_df[all_slice_df["StudyInstanceUID"] == study_id].reset_index(drop=True)
    new_df.append(one_study[x0:x1])
new_df = pd.concat(new_df, axis=0).reset_index(drop=True)
new_df  # 所有包含vertebra的slice



# %% [code] {"execution":{"iopub.status.busy":"2023-07-20T15:17:00.060474Z","iopub.status.idle":"2023-07-20T15:17:00.061139Z","shell.execute_reply.started":"2023-07-20T15:17:00.060853Z","shell.execute_reply":"2023-07-20T15:17:00.060880Z"}}
new_df = new_df.merge(voxel_crop_df, on="StudyInstanceUID", how="left")  # merge study_crop_df
assert len(slice_class_df) == len(new_df)

# %% [markdown]
# ### slice_class_df

# %% [code] {"execution":{"iopub.status.busy":"2023-07-20T15:17:00.063169Z","iopub.status.idle":"2023-07-20T15:17:00.063583Z","shell.execute_reply.started":"2023-07-20T15:17:00.063355Z","shell.execute_reply":"2023-07-20T15:17:00.063372Z"}}
new_slice_df = pd.concat([new_df, slice_class_df[["new_slice_num", "vertebra_class"]]], axis=1)
new_slice_df  # 合并 class

# %% [markdown]
# ### merge train.csv

# %% [code] {"execution":{"iopub.status.busy":"2023-07-20T15:17:00.066306Z","iopub.status.idle":"2023-07-20T15:17:00.067029Z","shell.execute_reply.started":"2023-07-20T15:17:00.066786Z","shell.execute_reply":"2023-07-20T15:17:00.066809Z"}}
tr_df = pd.read_csv(f"{datadir}/train.csv")
new_slice_df1 = new_slice_df.merge(tr_df, on="StudyInstanceUID", how="left")
new_slice_df1  # 合并 train.csv

# %% [code] {"execution":{"iopub.status.busy":"2023-07-20T15:17:00.068419Z","iopub.status.idle":"2023-07-20T15:17:00.069281Z","shell.execute_reply.started":"2023-07-20T15:17:00.068985Z","shell.execute_reply":"2023-07-20T15:17:00.069012Z"}}
new_slice_df1.to_csv(f"{datadir}/train_slice.csv", index=False)



# %% [markdown]
# ### vertebra level

# %% [code] {"execution":{"iopub.status.busy":"2023-07-20T15:17:00.070916Z","iopub.status.idle":"2023-07-20T15:17:00.071746Z","shell.execute_reply.started":"2023-07-20T15:17:00.071465Z","shell.execute_reply":"2023-07-20T15:17:00.071489Z"}}
sample_num = 24
vertebrae_df_list = []
for study_id in tqdm(np.unique(new_slice_df1["StudyInstanceUID"])):
    one_study = new_slice_df1[new_slice_df1["StudyInstanceUID"] == study_id].reset_index(drop=True)
    for cid in range(1, 8):
        one_study_cid = one_study[one_study["vertebra_class"] == cid].reset_index(drop=True)
        if len(one_study_cid) >= sample_num:
            sample_index = np.linspace(0, len(one_study_cid) - 1, sample_num, dtype=int)
            one_study_cid = one_study_cid.iloc[sample_index].reset_index(drop=True)
        if len(one_study_cid) < 1:
            continue
        slice_num_list = one_study_cid["slice_num"].values.tolist()
        arow = one_study_cid.iloc[0]
        vertebrae_df_list.append([f"{study_id}_{cid}", study_id, cid, slice_num_list, arow["before_image_size"], \
                                  arow["x0"], arow["x1"], arow["y0"], arow["y1"], arow["z0"], arow["z1"],
                                  arow[f"C{cid}"]])



# %% [code] {"execution":{"iopub.status.busy":"2023-07-20T15:17:00.073236Z","iopub.status.idle":"2023-07-20T15:17:00.074054Z","shell.execute_reply.started":"2023-07-20T15:17:00.073785Z","shell.execute_reply":"2023-07-20T15:17:00.073810Z"}}
len(vertebrae_df_list) / (2019 * 7)

# %% [code] {"execution":{"iopub.status.busy":"2023-07-20T15:17:00.075456Z","iopub.status.idle":"2023-07-20T15:17:00.076132Z","shell.execute_reply.started":"2023-07-20T15:17:00.075896Z","shell.execute_reply":"2023-07-20T15:17:00.075919Z"}}
vertebrae_df = pd.DataFrame(vertebrae_df_list, columns=["study_cid", "StudyInstanceUID", "cid", "slice_num_list", \
    "before_image_size", "x0", "x1", "y0", "y1", "z0", "z1", "label"])
vertebrae_df.to_pickle(f"{datadir}/vertebrae_df.pkl")
# vertebrae_df = pd.read_pickle(f"{datadir}/vertebrae_df.pkl")
# vertebrae_df

# %% [code]


# %% [markdown]
# ### study level

# %% [code] {"execution":{"iopub.status.busy":"2023-07-20T15:17:00.077744Z","iopub.status.idle":"2023-07-20T15:17:00.078811Z","shell.execute_reply.started":"2023-07-20T15:17:00.078377Z","shell.execute_reply":"2023-07-20T15:17:00.078411Z"}}
sample_num = 90
study_df_list = []
for study_id in tqdm(np.unique(new_slice_df1["StudyInstanceUID"])):
    one_study = new_slice_df1[new_slice_df1["StudyInstanceUID"] == study_id].reset_index(drop=True)
    if len(one_study) >= sample_num:
        sample_index = np.linspace(0, len(one_study) - 1, sample_num, dtype=int)
        one_study = one_study.iloc[sample_index].reset_index(drop=True)
    slice_num_list = one_study["slice_num"].values.tolist()
    arow = one_study.iloc[0]
    study_df_list.append(
        [study_id, slice_num_list, arow["before_image_size"], arow["x0"], arow["x1"], arow["y0"], arow["y1"],
         arow["z0"], arow["z1"], arow["patient_overall"]])


# %% [code] {"execution":{"iopub.status.busy":"2023-07-20T15:17:00.083804Z","iopub.status.idle":"2023-07-20T15:17:00.084886Z","shell.execute_reply.started":"2023-07-20T15:17:00.084563Z","shell.execute_reply":"2023-07-20T15:17:00.084594Z"}}
study_df = pd.DataFrame(study_df_list,
                        columns=["StudyInstanceUID", "slice_num_list", "before_image_size", "x0", "x1", "y0", "y1",
                                 "z0", "z1", "label"])
study_df.to_pickle(f"{datadir}/study_df_{sample_num}.pkl")

# study_df = pd.read_pickle(f"{datadir}/study_df_90.pkl")
study_df

