# %% [markdown]
# ## 1. Imports

# %%
import os
import gc
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
from PIL import Image
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
import sys
sys.path.append("/home/br/workspace/RSNA2023/input/pytorch-image-models-main/")
import timm
from timm import create_model, list_models
from timm.data import create_transform
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import random
import wandb
from wandb import AlertLevel
import torchvision
from torch.utils.data import Dataset
from torch.nn import functional as F
from torch.nn import Module, Linear, Sequential, ModuleList, ReLU, Dropout, Flatten
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD, AdamW, lr_scheduler
import warnings
warnings.filterwarnings('ignore')

gc.collect()
torch.cuda.empty_cache()

from utils import seed_everything, optimal_f1
from model import GeM



# %%
class CFG:
    image_size =  (1536, 896)
    tta = True
    
    seed = 1788
    num_workers = 5
    valid_batch_size = 16
    gpu_parallel = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    df_path = f'./input/vindr/vindr.csv'

    normalize_mean= [0.485, 0.456, 0.406]  # [0.21596, 0.21596, 0.21596]
    normalize_std = [0.229, 0.224, 0.225]  # [0.18558, 0.18558, 0.18558]

    target = 'cancer'
    ensemble = True


images_dir = f'./input/images_gpugen/vindr_1536896_16bit_cutoff'
output_dir = './input/vindr'

# %%
seed_everything(CFG.seed)

# %% [markdown]
# ## 2. Loading train/eval/test dataframes

# %%

df_train = pd.read_csv(CFG.df_path)

# %% [markdown]
# ## 3. Dataset class

# %%
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, 
    CenterCrop, Resize, RandomCrop, GaussianBlur, JpegCompression, Downscale, ElasticTransform, Affine, ToFloat
)
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(*, data):
    if data == 'train':
        return Compose([
            Normalize(mean=CFG.normalize_mean, std=CFG.normalize_std,),
            ToTensorV2(),
            ])
        
    elif data == 'valid':
        return Compose([
            ToFloat(max_value=65535.0),
            Resize(CFG.image_size[0], CFG.image_size[1]),
            ToTensorV2(),
        ])


# %%
class BreastCancerDataSet(Dataset):
    def __init__(self, df, path, transforms=None):
        super().__init__()
        self.df = df
        self.path = path
        self.transforms = transforms

    def __getitem__(self, i):
        path = f'{self.path}/{self.df.iloc[i].patient_id}/{self.df.iloc[i].image_id}.png'
        try:
            # img = Image.open(path).convert('RGB')
            # 16bit
            img = Image.open(path)
            img = np.array(img).astype(np.uint16)
            if img.ndim == 2:
                img = np.repeat(img[:, :, np.newaxis], 3, axis=-1)
        except Exception as ex:
            print(path, ex)
            return None

        if self.transforms:
            img = self.transforms(image=np.array(img))["image"]

        if CFG.target in self.df.columns:
            cancer_target = torch.as_tensor(self.df.iloc[i].cancer)
            return img, cancer_target

        return img

    def __len__(self):
        return len(self.df)


# %% [markdown]
# ## 3. Model

# %%


class BreastCancerModel(Module):
    def __init__(self, model_arch, dropout=0.):
        super().__init__()
        self.model = create_model(
            model_arch, 
            pretrained=True, 
            num_classes=0, 
            drop_rate=dropout,
            global_pool="", 
            )
        self.num_feats = self.model.num_features

        self.cancer_logits = Linear(self.num_feats, 1)
        
        self.fc_dropout = nn.Dropout(0)
        self.global_pool = GeM(p_trainable=True)

    def forward(self, x):
        x = self.model(x) # (bs, num_feats) /  (bs, num_feats, 16, 16)
        x = self.global_pool(x) # (bs, num_feats, 1, 1)
        x = x[:,:,0,0] # # (bs, num_feats)
        cancer_logits = self.cancer_logits(self.fc_dropout(x)).squeeze() # (bs)
        return cancer_logits

# %% [markdown]
# ## 4. Train: training/evaluation loop

# %%
def test_func(model, dataloader):
    model = model.to(CFG.device)
    cancer_pred_list = []
    with torch.no_grad():
        model.eval()
        for X in tqdm(dataloader, desc='Preds'):
            with autocast(enabled=True):
                X = X.to(CFG.device)
                pred_c = model(X).view(-1)
                if CFG.tta:
                    pred_c2 = model(torch.flip(X, dims=[-1])).view(-1) # horizontal mirror
                    pred_c = (pred_c + pred_c2) / 2
                
                cancer_pred_list.append(torch.sigmoid(pred_c))
        
        pred = torch.concat(cancer_pred_list).cpu().numpy()
        return pred

# %% [markdown]
# ## 6. Cross-validation

# %%
def gen_predictions(models, df_train, folds):
    df_train_predictions = []
    for model, fold in zip(models, folds):
        ds_valid = BreastCancerDataSet(df_train.query('split == @fold'), images_dir, get_transforms(data="valid"))
        valid_dataloader  = DataLoader(ds_valid, batch_size=CFG.valid_batch_size, shuffle=False, num_workers=CFG.num_workers, pin_memory=False)
        pred_cancer = test_func(model, valid_dataloader)
        print(f'Test fold:{fold}')
        
        df_pred = pd.DataFrame(data=pred_cancer, columns=['cancer_pred_proba'])

        df = pd.concat(
            [df_train.query('split == @fold').reset_index(drop=True), df_pred],
            axis=1
        ).sort_values(['patient_id', 'image_id'])
        df_train_predictions.append(df)
    df_train_predictions = pd.concat(df_train_predictions)
    return df_train_predictions


def gen_predictions_ensemble(models, df_train):
    df_pred_all = []
    for model in models:
        ds_valid = BreastCancerDataSet(df_train, images_dir, get_transforms(data="valid"))
        valid_dataloader  = DataLoader(ds_valid, batch_size=CFG.valid_batch_size, shuffle=False, num_workers=CFG.num_workers, pin_memory=False)
        pred_cancer = test_func(model, valid_dataloader)
        
        df_pred_all.append(pred_cancer)
    
    # mean of predictions
    df_preds = np.array(df_pred_all).mean(axis=0)
    
    df_preds = pd.DataFrame(data=df_preds, columns=['cancer_pred_proba'])
    df = pd.concat(
        [df_train, df_preds],
        axis=1
    ).sort_values(['patient_id', 'image_id'])
        
    return df



def load_model(path, backbone, model=None):
    state_dict = torch.load(path, map_location=CFG.device)
    if model is None:
        model = BreastCancerModel(backbone)
    model.load_state_dict(state_dict['model'])

    print(f"load model:{backbone}, thres:{state_dict['threshold']}, ")
    return model, state_dict['threshold'], state_dict['model_arch']

model_paths = [
        "./output/0001/0001_model_f0_ep10.pth", 
        "./output/0001/0001_model_f1_ep12.pth", 
        "./output/0001/0001_model_f2_ep7.pth", 
        "./output/0001/0001_model_f3_ep9.pth", 
        "./output/0001/0001_model_f4_ep8.pth", 
]

backbones = [
    "tf_efficientnetv2_s",
    "tf_efficientnetv2_s",
    "tf_efficientnetv2_s",
    "tf_efficientnetv2_s",
    "tf_efficientnetv2_s",
]

folds = [0,1,2,3,4]

assert len(model_paths) == len(folds) == len(backbones), f"got folds:{len(folds)}, model_paths:{len(model_paths)},  backbones:{len(backbones)}"


models = []

for m_path, backbone in zip(model_paths, backbones):
    model, thres, model_arch = load_model(m_path, backbone)
    model = model.to(CFG.device)
    models.append(model)
    print(f'm_path:{m_path}, model_arch:{model_arch}, thres:{thres}')

if CFG.ensemble:
    df_pred = gen_predictions_ensemble(models, df_train)
else:
    df_pred = gen_predictions(models, df_train, folds)

df_pred.to_csv(f'{output_dir}/train_predictions_0001.csv', index=False)



print(f'F1 CV score (multiple thresholds): {sklearn.metrics.f1_score(df_pred["cancer"], df_pred["cancer_pred"])}')



df_pred_all = df_pred.groupby(['patient_id', 'laterality']).agg(
    cancer_max=('cancer_pred_proba', 'max'), cancer_mean=('cancer_pred_proba', 'mean'), cancer=('cancer', 'max')
)
print(f'ALL pF1 CV score. Mean , single threshold: {optimal_f1(df_pred_all["cancer"].values, df_pred_all["cancer_mean"].values)}', )
print(f'ALL pF1 CV score. Max  , single threshold: {optimal_f1(df_pred_all["cancer"].values, df_pred_all["cancer_max"].values)}', )

df_pred1 = df_pred[df_pred["site_id"]==1].groupby(['patient_id', 'laterality']).agg(
    cancer_max=('cancer_pred_proba', 'max'), cancer_mean=('cancer_pred_proba', 'mean'), cancer=('cancer', 'max')
)
print(f'SITE1 pF1 CV score. Mean , single threshold: {optimal_f1(df_pred1["cancer"].values, df_pred1["cancer_mean"].values)}', )
print(f'SITE1 pF1 CV score. Max  , single threshold: {optimal_f1(df_pred1["cancer"].values, df_pred1["cancer_max"].values)}', )

df_pred2 = df_pred[df_pred["site_id"]==2].groupby(['patient_id', 'laterality']).agg(
    cancer_max=('cancer_pred_proba', 'max'), cancer_mean=('cancer_pred_proba', 'mean'), cancer=('cancer', 'max')
)
print(f'SITE2 pF1 CV score. Mean , single threshold: {optimal_f1(df_pred2["cancer"].values, df_pred2["cancer_mean"].values)}', )
print(f'SITE2 pF1 CV score. Max  , single threshold: {optimal_f1(df_pred2["cancer"].values, df_pred2["cancer_max"].values)}', )

