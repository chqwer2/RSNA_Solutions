# %% [markdown] {"id":"Ao5etbwLxNED"}
# # Profile

# %% [code] {"executionInfo":{"elapsed":200845,"status":"ok","timestamp":1615451946014,"user":{"displayName":"徐铭","photoUrl":"https://lh3.googleusercontent.com/a-/AOh14GiblzBIU0N3bzEW-w5kOCgqAoahWcHJnTU5mI-A=s64","userId":"11772050295087096956"},"user_tz":-480},"id":"3Bf8PD-jPkYa"}
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
datadir = '../../rsna_cervical_spine'

libdir = '.'
outputdir = '.'
otherdir = '.'

train_bs_ = 16  # train_batch_size
valid_bs_ = 128  # valid_batch_size
num_workers_ = 5


# %% [markdown] {"id":"qu0pJE-UxU7u"}
# # CFG

# %% [code] {"executionInfo":{"elapsed":200843,"status":"ok","timestamp":1615451946014,"user":{"displayName":"徐铭","photoUrl":"https://lh3.googleusercontent.com/a-/AOh14GiblzBIU0N3bzEW-w5kOCgqAoahWcHJnTU5mI-A=s64","userId":"11772050295087096956"},"user_tz":-480},"id":"6fBgDehlNAAl"}
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
    img_size = 320
    croped_img_size = 320  # 裁剪后的图片尺寸
    weight_path = f"{outputdir}/efficientnet-b0_109_fold0_epoch13.pth"


# %% [markdown] {"id":"p4w3IC-Qvcyq"}
# # Import

# %% [code] {"executionInfo":{"elapsed":222046,"status":"ok","timestamp":1615451967224,"user":{"displayName":"徐铭","photoUrl":"https://lh3.googleusercontent.com/a-/AOh14GiblzBIU0N3bzEW-w5kOCgqAoahWcHJnTU5mI-A=s64","userId":"11772050295087096956"},"user_tz":-480},"id":"4U1yFOTSM5Jw","outputId":"5e7b5f5c-88ed-40ff-f025-4c28bff1fd43"}
# !pip install -q git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git

import sys;

package_paths = [f'{libdir}pytorch-image-models-master']
for pth in package_paths:
    sys.path.append(pth)

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

if CFG.device == 'TPU':
    # !pip
    # install - q
    # pytorch - ignite
    import ignite.distributed as idist
elif CFG.device == 'GPU':
    from torch.cuda.amp import autocast, GradScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# %% [code] {"executionInfo":{"elapsed":225887,"status":"ok","timestamp":1615451971076,"user":{"displayName":"徐铭","photoUrl":"https://lh3.googleusercontent.com/a-/AOh14GiblzBIU0N3bzEW-w5kOCgqAoahWcHJnTU5mI-A=s64","userId":"11772050295087096956"},"user_tz":-480},"id":"8X-g-a_hM_zN"}
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


# %% [markdown] {"id":"g6YIAKXXe81-"}
# # 87Sampler Peek Mask

# %% [code]
# study_train_df = pd.read_csv(f'{datadir}/seg_25d.csv')
# print('train_df shape:', study_train_df.shape)
# study_train_df.head(3)

# %% [code]
# import segmentation_models_pytorch as smp

# def build_model():
#     model = smp.Unet(
#         encoder_name=CFG.model_arch,    # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#         encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#         in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#         classes=CFG.num_classes,        # model output channels (number of classes in your dataset)
#         activation=None,
#     )
#     model.to(device)
#     return model

# def load_model(path):
#     model = build_model()
#     model.load_state_dict(torch.load(path)["model"])
#     model.eval()
#     return model

# model = load_model(CFG.weight_path)
# model.eval()
# gc.collect()

# %% [code]
# import random
# rand_idx = random.randint(0,len(study_train_df))
# print(f"rand_idx:{rand_idx}")
# example = study_train_df.iloc[rand_idx]
# exa_image = np.load(example["image_path"])
# exa_mask = np.load(example["mask_path"])
# print(f"exa_image.shape: {exa_image.shape}, exa_mask.shape: {exa_mask.shape}")

# exa_image = np.expand_dims((exa_image/255).transpose(2,0,1), 0)
# exa_image = torch.from_numpy(exa_image)
# exa_image = exa_image.to(device, dtype=torch.float)
# with torch.no_grad():
#     y_pred = model(exa_image)
# y_pred = y_pred.sigmoid() ####
# y_pred = (y_pred).to('cpu').numpy()
# slice_pred = y_pred[0] # 8 * img_size * img_size
# slice_mask_argmax = slice_pred.argmax(0) # img_size * img_size
# slice_mask_max = slice_pred.max(0) # img_size * img_size
# slice_mask = np.where(slice_mask_max>0.5, slice_mask_argmax, 0).astype(np.uint8)

# %% [code]
# from matplotlib.patches import Rectangle

# plt.figure(figsize=(30, 20))
# img = exa_image[0][1].cpu().numpy() # 512*512; 0-1;
# slice_mask = slice_mask # 512*512; 0or1;
# label_mask = exa_mask[:,:,1].astype("uint8") # 512*512; 0-8 classes;
# plt.subplot(1, 3, 1); plt.imshow(img); plt.axis('OFF'); # 512*512; 0-1;
# plt.subplot(1, 3, 2); plt.imshow(slice_mask); plt.axis('OFF');
# plt.subplot(1, 3, 3);
# plt.imshow(exa_mask[:,:,1]);
# plt.axis('OFF');
# # plt.subplot(1, 3, 3); plt.imshow(slice_mask); plt.imshow(img,alpha=0.7); plt.axis('OFF');
# # # plt.colorbar()
# plt.tight_layout()
# plt.show()

# %% [markdown]
# # 2019Train CSV

# %% [code]
study_train_df = pd.read_csv(f'{datadir}/train.csv')
print('train_df shape:', study_train_df.shape)
study_train_df.head(3)


seg_paths = glob(f"{datadir}/segmentations/*")
seg_gt_list = [path.split('/')[-1][:-4] for path in seg_paths]


# %% [code] {"execution":{"iopub.status.busy":"2023-07-20T15:16:19.601025Z","iopub.execute_input":"2023-07-20T15:16:19.601734Z","iopub.status.idle":"2023-07-20T15:16:19.615692Z","shell.execute_reply.started":"2023-07-20T15:16:19.601693Z","shell.execute_reply":"2023-07-20T15:16:19.614176Z"}}
study_train_df = study_train_df[~study_train_df["StudyInstanceUID"].isin(seg_gt_list)]



# %% [code]
train_slice_list = []
for file_name in tqdm(study_train_df["StudyInstanceUID"].values):
    train_image_path = glob(f"{datadir}/train_images/{file_name}/*")
    train_image_path = sorted(train_image_path, key=lambda x: int(x.split("/")[-1].replace(".dcm", "")))
    for path_idx in range(len(train_image_path)):
        path1 = "nofile" if path_idx - 1 < 0 else train_image_path[path_idx - 1].replace(f"{datadir}/", "")
        path2 = train_image_path[path_idx].replace(f"{datadir}/", "")
        path3 = "nofile" if path_idx + 1 >= len(train_image_path) else train_image_path[path_idx + 1].replace(
            f"{datadir}/", "")
        slice_num = int(path2.split("/")[-1].replace(".dcm", ""))
        train_slice_list.append([f"{file_name}_{slice_num}", file_name, slice_num, path1, path2, path3])

# %% [code]
train_df = pd.DataFrame(train_slice_list, columns=["id", "StudyInstanceUID", "slice_num", "path1", "path2", "path3"])
train_df = train_df.sort_values(['StudyInstanceUID', 'slice_num'], ascending=[True, True]).reset_index(drop=True)
train_df.to_csv(f'{datadir}/train_slice_list.csv', index=False)
train_df


# %% [markdown] {"id":"RXBaZcUpQLqw"}
# # Dataset

# %% [code] {"executionInfo":{"elapsed":225885,"status":"ok","timestamp":1615451971076,"user":{"displayName":"徐铭","photoUrl":"https://lh3.googleusercontent.com/a-/AOh14GiblzBIU0N3bzEW-w5kOCgqAoahWcHJnTU5mI-A=s64","userId":"11772050295087096956"},"user_tz":-480},"id":"fuxNse_1M_r0"}
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

    # %% [code] {"executionInfo":{"elapsed":225884,"status":"ok","timestamp":1615451971077,"user":{"displayName":"徐铭","photoUrl":"https://lh3.googleusercontent.com/a-/AOh14GiblzBIU0N3bzEW-w5kOCgqAoahWcHJnTU5mI-A=s64","userId":"11772050295087096956"},"user_tz":-480},"id":"kGjOdUreM_jF"}


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


# %% [code] {"executionInfo":{"elapsed":229121,"status":"ok","timestamp":1615451974322,"user":{"displayName":"徐铭","photoUrl":"https://lh3.googleusercontent.com/a-/AOh14GiblzBIU0N3bzEW-w5kOCgqAoahWcHJnTU5mI-A=s64","userId":"11772050295087096956"},"user_tz":-480},"id":"h_AG-OSlBRm5","outputId":"10f6d8fa-36f1-4612-ef86-7050ae144390"}
from pylab import rcParams

dataset_show = TrainDataset(
    train_df,
    get_transforms("valid")  # None, get_transforms("train")
)
rcParams['figure.figsize'] = 30, 20
for i in range(2):
    f, axarr = plt.subplots(1, 3)
    idx = np.random.randint(0, len(dataset_show))
    img, file_name, n_slice = dataset_show[idx]
    # axarr[p].imshow(img) # transform=None
    axarr[0].imshow(img[0]);
    plt.axis('OFF');
    axarr[1].imshow(img[1]);
    plt.axis('OFF');
    axarr[2].imshow(img[2]);
    plt.axis('OFF');

# %% [markdown] {"id":"GbWy6sEbRl8P"}
# # Model

# %% [code] {"executionInfo":{"elapsed":229120,"status":"ok","timestamp":1615451974323,"user":{"displayName":"徐铭","photoUrl":"https://lh3.googleusercontent.com/a-/AOh14GiblzBIU0N3bzEW-w5kOCgqAoahWcHJnTU5mI-A=s64","userId":"11772050295087096956"},"user_tz":-480},"id":"ty22GW9qRkDk"}
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

# %% [code]
# os.makedirs(f"{outputdir}/train_voxel", exist_ok=True)
os.makedirs(f"{outputdir}/train_voxel_mask", exist_ok=True)
# for filename in train_df["StudyInstanceUID"].values:
#     os.makedirs(f"{outputdir}/train_mask/{filename}", exist_ok=True)

# %% [code]
vertebra_class = []
voxel_crop_list = []


def crop_voxel(voxels, voxel_mask, last_f_name, croped_img_size):
    area_thr = 10
    # x
    x_list = []
    length = voxel_mask.shape[0]
    for i in range(length):
        if voxel_mask[i].sum().item() >= area_thr:
            x_list.append(i)
            break
    else:
        x_list.append(0)

    for i in range(length - 1, -1, -1):
        if voxel_mask[i].sum().item() >= area_thr:
            x_list.append(i)
            break
    else:
        x_list.append(length - 1)

    # y
    y_list = []
    length = voxel_mask.shape[1]
    for i in range(length):
        if voxel_mask[:, i].sum().item() >= area_thr:
            y_list.append(i)
            break
    else:
        y_list.append(0)

    for i in range(length - 1, -1, -1):
        if voxel_mask[:, i].sum().item() >= area_thr:
            y_list.append(i)
            break
    else:
        y_list.append(length - 1)

    # z
    z_list = []
    length = voxel_mask.shape[2]
    for i in range(length):
        if voxel_mask[:, :, i].sum().item() >= area_thr:
            z_list.append(i)
            break
    else:
        z_list.append(0)

    for i in range(length - 1, -1, -1):
        if voxel_mask[:, :, i].sum().item() >= area_thr:
            z_list.append(i)
            break
    else:
        z_list.append(length - 1)
    # croped_voxel = voxels[x_list[0]:x_list[1]+1, y_list[0]:y_list[1]+1, z_list[0]:z_list[1]+1]
    try:
        croped_voxel_mask = voxel_mask[x_list[0]:x_list[1] + 1, y_list[0]:y_list[1] + 1, z_list[0]:z_list[1] + 1]
        row = [last_f_name, voxel_mask.shape[1], x_list[0], x_list[1] + 1, y_list[0], y_list[1] + 1, z_list[0],
               z_list[1] + 1]
    except:
        print(
            f"last_f_name:{last_f_name}, voxel_mask.shape:{voxel_mask.shape}, x_list:{x_list}, y_list:{y_list}, z_list:{z_list}")
        croped_voxel_mask = voxel_mask
        row = [last_f_name, voxel_mask.shape[1], 0, voxel_mask.shape[0], 0, voxel_mask.shape[1], 0, voxel_mask.shape[2]]
    voxel_crop_list.append(row)

    # croped_voxel = croped_voxel.to('cpu').numpy() # bs*img_size*img_size; 0-8 classes
    croped_voxel_mask = croped_voxel_mask.to('cpu').numpy().astype(np.uint8)  # bs*img_size*img_size; 0-8 classes
    # print(voxel_mask.shape, "to1", croped_voxel_mask.shape)
    # # resize
    # croped_voxel_list = []
    # croped_voxel_mask_list = []

    for x_idx in range(croped_voxel_mask.shape[0]):
        # slice = croped_voxel[x_idx]
        slice_mask = croped_voxel_mask[x_idx]

        unique, counts = np.unique(slice_mask, return_counts=True)
        if len(unique) == 1 and unique[0] == 0:
            vertebra_class.append([last_f_name, x_idx, 0])
        elif unique[0] == 0:
            unique = unique[1:]
            counts = counts[1:]
            vertebra_class.append([last_f_name, x_idx, unique[counts.argmax()]])
        else:
            vertebra_class.append([last_f_name, x_idx, unique[counts.argmax()]])

        # croped_voxel_list.append(cv2.resize(slice, (image_size, image_size), interpolation=cv2.INTER_NEAREST))
        # croped_voxel_mask_list.append(cv2.resize(slice_mask, (croped_img_size, croped_img_size), interpolation=cv2.INTER_NEAREST))

    # croped_voxel = np.array(croped_voxel_list)
    # croped_voxel_mask = np.array(croped_voxel_mask_list)

    # np.save(f"{outputdir}/train_voxel/{last_f_name}.npy", croped_voxel)
    # np.save(f"{outputdir}/train_voxel_mask/{last_f_name}.npy", croped_voxel_mask)

    # print(voxel_mask.shape, "to2", croped_voxel_mask.shape)
    # np.save(f"{outputdir}/train_mask/{last_f_name}.npy", voxel_mask)
    return None, croped_voxel_mask


# %% [code] {"executionInfo":{"elapsed":229495,"status":"ok","timestamp":1615451974706,"user":{"displayName":"徐铭","photoUrl":"https://lh3.googleusercontent.com/a-/AOh14GiblzBIU0N3bzEW-w5kOCgqAoahWcHJnTU5mI-A=s64","userId":"11772050295087096956"},"user_tz":-480},"id":"fchpKK8dAWQU"}
for fold in range(CFG.fold_num):
    if fold in CFG.fold_list:
        test_dataset = TrainDataset(train_df, transform=get_transforms("valid"))  # get_transforms("valid")
        test_loader = DataLoader(test_dataset, batch_size=CFG.valid_bs, shuffle=False, num_workers=CFG.num_workers,
                                 pin_memory=True, drop_last=False)

        model = load_model(CFG.weight_path)
        model.eval()
        last_f_name = ""
        voxel_mask = []
        voxels = []
        for step, (images, file_names, n_slice) in tqdm(enumerate(test_loader), total=len(test_loader)):
            images = images.to(device, dtype=torch.float)  # bs*3*image_size*image_size
            batch_size = images.size(0)
            with torch.no_grad():
                y_pred = model(images)  # [B, 8, H, W]
            y_pred = y_pred.sigmoid()  ####
            slice_mask_max = torch.max(y_pred, 1)  # bs*img_size*img_size
            slice_mask = torch.where((slice_mask_max.values) > 0.5, slice_mask_max.indices + 1,
                                     0)  # bs*img_size*img_size; 0-8 classes
            # slice_mask = slice_mask.to('cpu').numpy().astype(np.uint8) # bs*img_size*img_size; 0-8 classes

            # slice_image = images[:, 1, :, :] # bs*img_size*img_size
            start_idx = 0
            for bs_idx in range(batch_size):
                f_name = file_names[bs_idx]
                if f_name != last_f_name:
                    voxel_mask.append(slice_mask[start_idx:bs_idx])
                    # voxels.append(slice_image[start_idx:bs_idx])
                    voxel_mask = torch.cat(voxel_mask, dim=0)  # n_slice*img_size*img_size; 0-8 classes
                    # voxels = torch.cat(voxels, dim=0) # n_slice*img_size*img_size
                    if len(voxel_mask) > 0:
                        croped_voxel, croped_voxel_mask = crop_voxel(None, voxel_mask, last_f_name, CFG.croped_img_size)
                    last_f_name = f_name
                    start_idx = bs_idx
                    voxel_mask = []
                    # voxels = []
                elif bs_idx == batch_size - 1:
                    voxel_mask.append(slice_mask[start_idx:batch_size])
                    # voxels.append(slice_image[start_idx:batch_size])
        voxel_mask = torch.cat(voxel_mask, dim=0)
        if len(voxel_mask) > 0:
            croped_voxel, croped_voxel_mask = crop_voxel(None, voxel_mask, last_f_name, CFG.croped_img_size)
        torch.cuda.empty_cache()
        gc.collect()


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
# ### voxel_crop_list

# %% [code]
len(voxel_crop_list)  # 2019

# %% [code]
study_crop_df = pd.DataFrame(voxel_crop_list,
                             columns=["StudyInstanceUID", "before_image_size", "x0", "x1", "y0", "y1", "z0", "z1"])
study_crop_df.to_csv(f"{datadir}/study_crop_info.csv", index=False)
study_crop_df

# %% [code]
train_slice_list = pd.read_csv(f"{datadir}/train_slice_list.csv")
train_slice_list

# %% [code]
new_df = []
for idx, study_id, _, x0, x1, _, _, _, _, in study_crop_df.itertuples():
    one_study = train_slice_list[train_slice_list["StudyInstanceUID"] == study_id].reset_index(drop=True)
    new_df.append(one_study[x0:x1])
new_df = pd.concat(new_df, axis=0).reset_index(drop=True)
new_df

# %% [markdown]
# ### slice_class_df

# %% [code]
slice_class_df = pd.DataFrame(vertebra_class, columns=["StudyInstanceUID", "new_slice_num", "vertebra_class"])
slice_class_df.sort_values(by=["StudyInstanceUID", "new_slice_num"], inplace=True)
slice_class_df

# %% [code]
new_df = pd.concat([new_df, slice_class_df[["new_slice_num", "vertebra_class"]]], axis=1)
new_df

# %% [code]
new_slice_df = new_df.merge(study_crop_df, on="StudyInstanceUID", how="left")
new_slice_df

# %% [markdown]
# ### merge train.csv

# %% [code]
tr_df = pd.read_csv(f"{datadir}/train.csv")
tr_df

# %% [code]
new_slice_df1 = new_slice_df.merge(tr_df, on="StudyInstanceUID", how="left")
new_slice_df1

# %% [code]
new_slice_df1.to_csv(f"{datadir}/train_slice.csv", index=False)

# %% [code]
sample_num = 24
vertebrae_df_list = []
for study_id in tqdm(np.unique(new_slice_df1["StudyInstanceUID"])):
    one_study = new_slice_df1[new_slice_df1["StudyInstanceUID"] == study_id].reset_index(drop=True)
    for cid in range(1, 8):
        one_study_cid = one_study[one_study["vertebra_class"] == cid].reset_index(drop=True)
        if len(one_study_cid) >= sample_num:
            sample_index = np.linspace(0, len(one_study_cid) - 1, sample_num, dtype=int)
            one_study_cid = one_study_cid.iloc[sample_index].reset_index(drop=True)
        if len(one_study_cid) < 5:
            continue
        slice_num_list = one_study_cid["slice_num"].values.tolist()
        arow = one_study_cid.iloc[0]
        vertebrae_df_list.append([f"{study_id}_{cid}", study_id, cid, slice_num_list, arow["before_image_size"], \
                                  arow["x0"], arow["x1"], arow["y0"], arow["y1"], arow["z0"], arow["z1"],
                                  arow[f"C{cid}"]])

# %% [code]
vertebrae_df = pd.DataFrame(vertebrae_df_list, columns=["study_cid", "StudyInstanceUID", "cid", "slice_num_list", \
                                                        "before_image_size", "x0", "x1", "y0", "y1", "z0", "z1",
                                                        "label"])
vertebrae_df.to_pickle(f"{datadir}/vertebrae_df.pkl")
vertebrae_df

# %% [code]
# plt.figure(figsize=(30, 20))
# img = images[0].cpu().numpy().transpose(1,2,0)
# plt.subplot(1, 3, 1); plt.imshow(slice_mask); plt.axis('OFF');
# plt.subplot(1, 3, 2); plt.imshow(img); plt.axis('OFF');
# plt.subplot(1, 3, 3); plt.imshow(slice_mask); plt.imshow(img,alpha=0.7); plt.axis('OFF');
# # # plt.colorbar()
# plt.tight_layout()
# plt.show()

# %% [code]
