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


os.environ['WANDB_API_KEY'] = 'Your api key'
gc.collect()
torch.cuda.empty_cache()

from utils import seed_everything, init_logger, get_timediff, optimal_f1, gc_collect, add_weight_decay, get_parameter_number
from model import GeM, BreastCancerModel
from dataset import BreastCancerDataSet_16bit, BreastCancerDataSet_8bit, mixup_augmentation, get_transforms_8bit, get_transforms_16bit


# %%
class CFG:
    suff = "0003"
    image_size =  (1536, 896)
    epochs = 5
    model_arch = 'tf_efficientnetv2_s' # tf_efficientnetv2_s / convnextv2_tiny.fcmae_ft_in22k_in1k_384 / convnextv2_tiny
    dropout = 0.0
    fc_dropout=0.2
    es_paitient = 3

    onecycle = True
    onecycle_pct_start = 0.1
    max_lr = 1e-6
    optim = "AdamW"
    weight_decay = 0.01
    accum_iter=1

    positive_target_weight = 1
    neg_downsample = 0.35
    train_batch_size = 8
    valid_batch_size = 16
    mixup_rate = 0.5
    mixup_alpha = 0.5

    tta = True
    
    seed = 1788
    num_workers = 5
    n_folds = 5
    folds = [0]
    gpu_parallel = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    df_path = f'./input/df_train_0001.csv'

    checkpoint_path = f'./input/checkpoint/0002_effv2s_PL_f4_ep6.pth'

    normalize_mean= [0.485, 0.456, 0.406]  # [0.21596, 0.21596, 0.21596]
    normalize_std = [0.229, 0.224, 0.225]  # [0.18558, 0.18558, 0.18558]

    wandb_project = f'RSNA2023-V5' 
    wandb_run_name = f'{suff}_{model_arch}'

    target = 'cancer'


comp_data_dir = './input/rsna-breast-cancer-detection'
images_dir = f'./input/images_gpugen/1536896_16bit_cutoff'
output_dir = f'./output/{CFG.suff}'
os.makedirs(output_dir, exist_ok=True)

DEBUG = True
WANDB_SWEEP = False
TRAIN = True
CV = True

# %% [markdown]
# ## Helper Function

# %%
seed_everything(CFG.seed)

LOGGER = init_logger(f'{output_dir}/train_{CFG.suff}.log')

# %%
LOGGER.info(f'run: {CFG.wandb_run_name}; folds:{CFG.folds}')
LOGGER.info(f"timm.version: {timm.__version__}")
LOGGER.info(f"checkpoint: {CFG.checkpoint_path.split('/')[-1]}")

try:
    df_train = pd.read_csv(CFG.df_path)
except:
    LOGGER.info(f"Can't find {CFG.df_path}, creating one")
    df_train = pd.read_csv(f'{comp_data_dir}/train.csv')
    split = StratifiedGroupKFold(CFG.n_folds)
    for k, (_, test_idx) in enumerate(split.split(df_train, df_train.cancer, groups=df_train.patient_id)):
        df_train.loc[test_idx, 'split'] = k
    df_train.split = df_train.split.astype(int)
    df_train["sample_rand"] = np.random.rand(len(df_train)) 
    df_train.loc[df_train["cancer"]==1, "sample_rand"] = 0.0
    df_train.to_csv(f'./input/df_train_{CFG.suff}.csv', index=False)

df_train = df_train[df_train['image_id'].astype(str) != '1942326353'].reset_index(drop=True)
df_train = df_train[df_train['patient_id'].astype(str) != '27770'].reset_index(drop=True)

# %%
# 3. Dataset class

if DEBUG:
    ds_train = BreastCancerDataSet_16bit(df_train, images_dir, CFG.target, get_transforms_16bit('train', CFG.image_size, CFG.normalize_mean, CFG.normalize_std))
    X, y_cancer = ds_train[42]
    print(f"X.shape: {X.shape}, y_cancer.shape: {y_cancer.shape}")


dataset_show = BreastCancerDataSet_16bit(df_train, images_dir, CFG.target, get_transforms_16bit('train', CFG.image_size, CFG.normalize_mean, CFG.normalize_std))
for i in range(2):
    f, axarr = plt.subplots(1,3, figsize=(10,8))
    for p in range(0,3):
        idx = np.random.randint(0, len(dataset_show))
        img, cancer_target = dataset_show[idx]
        img = ((img-img.min())/(img.max()-img.min())*255).to(torch.uint8).transpose(0, 1).transpose(1,2)
        axarr[p].imshow(img)
        axarr[p].set_title(str(cancer_target.item()))
        axarr[p].axis('off')
        plt.tight_layout()
    plt.savefig(f'{output_dir}/show_{i}.jpg')
    # plt.show()


# 3. Model
if DEBUG:
    with torch.no_grad():
        model = BreastCancerModel(model_arch=CFG.model_arch, dropout=0.0, fc_dropout=0.0)
        pred = model(torch.randn(2, 3, 512, 512))
        print('model output:', pred.shape)
        LOGGER.info(get_parameter_number(model))
    del model

# %% [markdown]
# ## 4. Train: training/evaluation loop

# %%
def valid_one_epoch(model, dataloader):
    model = model.to(CFG.device)
    cancer_pred_list = []
    with torch.no_grad():
        model.eval()
        losses = []; targets = []
        with tqdm(dataloader, desc='Eval', mininterval=30) as progress:
            for i, (X, y_c) in enumerate(progress):
                with autocast(enabled=True):
                    X = X.to(CFG.device)
                    y_c = y_c.to(float).to(CFG.device)
                    
                    pred_c = model(X).view(-1)
                    if CFG.tta:
                        pred_c2 = model(torch.flip(X, dims=[-1])) # horizontal mirror
                        pred_c = (pred_c + pred_c2) / 2

                    loss = F.binary_cross_entropy_with_logits(pred_c, y_c, pos_weight=torch.tensor([CFG.positive_target_weight]).to(CFG.device)).item()
                    loss = loss / CFG.accum_iter
                    
                    cancer_pred_list.append(torch.sigmoid(pred_c))
                    losses.append(loss); targets.append(y_c.cpu().numpy())
        
        targets = np.concatenate(targets)
        pred = torch.concat(cancer_pred_list).cpu().numpy()
        pf1, thres = optimal_f1(targets, pred)
        #      (best_pf1, best_thres)  pred_value  mean_all_losses
        return (pf1,      thres),      pred,       np.mean(losses)

# %%
def train_one_epoch(model, dl, optim, scheduler, cancer_criterion, epoch, logger):
    model.train()
    scaler = GradScaler()
    losses = []
    with tqdm(dl, desc='Train', mininterval=10) as train_progress:
        for batch_idx, (img, yc) in enumerate(train_progress):
            img = img.to(CFG.device)
            yc = yc.to(float).to(CFG.device)
            
            # Mixup- allowed
            if torch.randn(1)[0] < CFG.mixup_rate and img.shape[0]>1:  
                mixed_x, yc_j, yc_k, lam = mixup_augmentation(img, yc, alpha=CFG.mixup_alpha)
                with autocast(enabled=True):
                    pred_c = model(mixed_x).view(-1) 
                    # Mixup loss calculation
                    loss_j = cancer_criterion(pred_c, yc_j, pos_weight=torch.tensor([CFG.positive_target_weight]).to(CFG.device)) 
                    loss_k = cancer_criterion(pred_c, yc_k, pos_weight=torch.tensor([CFG.positive_target_weight]).to(CFG.device))
                    loss = lam * loss_j + (1 - lam) * loss_k
        
            # Mixup - not allowed
            else:
                if img.shape[0] <= 1:
                    print('batch size is 1, skipping mixup') 
                with autocast(enabled=True):
                    pred_c = model(img).view(-1) 
                    loss = cancer_criterion(pred_c, yc, pos_weight=torch.tensor([CFG.positive_target_weight]).to(CFG.device)) 

            if np.isinf(loss.item()) or np.isnan(loss.item()):
                print(f'Bad loss, skipping the batch {batch_idx}')
                del loss, pred_c
                gc_collect()
                continue
            loss = loss / CFG.accum_iter
            losses.append(loss.item())
            scaler.scale(loss).backward() # scaler is needed to prevent "gradient underflow"
            if (batch_idx + 1) % CFG.accum_iter == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()

            if scheduler is not None:
                scheduler.step()
            
            logger.log({'tr_loss': (loss.item()),
                        'lr': scheduler.get_last_lr()[0] if scheduler else CFG.max_lr,
                        'epoch': epoch})
            
    return model, np.mean(losses[-30:])

# %%
def load_model(path, model=None):
    state_dict = torch.load(path, map_location=CFG.device)
    if model is None:
        model = BreastCancerModel(state_dict['model_arch'], CFG.dropout, CFG.fc_dropout)
    model.load_state_dict(state_dict['model'])
    return model, state_dict['threshold'], state_dict['model_arch']

def train_loop(logger, fold, do_save_model=True):
    # ====================================================
    # data loader
    # ====================================================
    tr_df = df_train.query('split != @fold')
    tr_df = tr_df[tr_df["sample_rand"] <= CFG.neg_downsample].reset_index(drop=True)
    va_df = df_train.query('split == @fold')
    LOGGER.info(f"train: {len(tr_df)}, train pos rate: {tr_df['cancer'].mean():.3f}")
    LOGGER.info(f"valid: {len(va_df)}, valid pos rate: {va_df['cancer'].mean():.3f}")

    ds_train = BreastCancerDataSet_16bit(tr_df, images_dir, CFG.target, get_transforms_16bit('train', CFG.image_size, CFG.normalize_mean, CFG.normalize_std))
    ds_valid = BreastCancerDataSet_16bit(va_df, images_dir, CFG.target, get_transforms_16bit('valid', CFG.image_size, CFG.normalize_mean, CFG.normalize_std))
    dl_train = DataLoader(ds_train, batch_size=CFG.train_batch_size, shuffle=True,  num_workers=CFG.num_workers, pin_memory=True,  drop_last=True)
    dl_valid = DataLoader(ds_valid, batch_size=CFG.valid_batch_size, shuffle=False, num_workers=CFG.num_workers, pin_memory=False, drop_last=False)
    
    # ====================================================
    # model & optimizer & scheduler & loss
    # ====================================================
    # model
    # model = BreastCancerModel(CFG.model_arch, CFG.dropout, CFG.fc_dropout).to(CFG.device)
    # if CFG.gpu_parallel:    
    #     from torch.nn import DataParallel
    #     num_gpu = torch.cuda.device_count()
    #     model = DataParallel(model, device_ids=range(num_gpu))
    #     LOGGER.info(f"enable gpu parallel, num_gpu: {num_gpu}")
    model = load_model(CFG.checkpoint_path)[0]
    model = model.to(CFG.device)

    # optimizer
    if CFG.optim == "AdamW":
        optim = AdamW(add_weight_decay(model, weight_decay=CFG.weight_decay, skip_list=['bias']), lr=CFG.max_lr, betas=(0.9, 0.999), weight_decay=CFG.weight_decay)
    elif CFG.optim == "Adam":
        optim = Adam(model.parameters())
    
    # scheduler
    scheduler = None
    if CFG.onecycle:
        scheduler = lr_scheduler.OneCycleLR(optim, max_lr=CFG.max_lr, epochs=CFG.epochs, steps_per_epoch=len(dl_train), pct_start=CFG.onecycle_pct_start)
    
    # loss
    cancer_criterion = F.binary_cross_entropy_with_logits

    # ====================================================
    # loop
    # ====================================================    
    best_valid_score = 0; best_valid_thres=0
    n_es = 0
    for epoch in range(CFG.epochs):
        model, tr_loss = train_one_epoch(model, dl_train, optim, scheduler, cancer_criterion, epoch, logger)
        (f1, thres), _, va_loss = valid_one_epoch(model, dl_valid)

        n_es += 1
        if f1 > best_valid_score:
            n_es = 0
            best_valid_score = f1
            best_valid_thres = thres
            if do_save_model and epoch > 0:
                save_name = f'{output_dir}/{CFG.suff}_{CFG.model_arch}_FT_f{fold}_ep{epoch}.pth'
                torch.save({'model': model.state_dict(), 'threshold': thres, 'model_arch': CFG.model_arch}, save_name)
                best_dict[fold] = save_name

        LOGGER.info(f'Epoch {epoch} - valid_f1: {f1:.4f} - valid_thres: {thres:.4f} - best_f1: {best_valid_score:.4f} - best_thres: {best_valid_thres:.4f}, lr: {scheduler.get_last_lr()[0] if scheduler else CFG.max_lr}')
        LOGGER.info(f"train_loss: {tr_loss:.4f} - valid_loss: {va_loss:.4f}")
        logger.log({
            'fold':       fold,
            'epoch':      epoch,
            'va_loss':    va_loss,
            'va_pf1':     f1,
            'va_thres':   thres,
            'best_pf1':   best_valid_score,
            'best_thres': best_valid_thres,
            })

        if n_es > CFG.es_paitient:
            LOGGER.info(f'Early Stopping - Epoch: {epoch}')
            break

# %%
if TRAIN:
    best_dict = {}
    for fold in CFG.folds:
        LOGGER.info(f'\n========== Fold {fold} ==========')
        with wandb.init(project=CFG.wandb_project, name=f'{CFG.wandb_run_name}-f{fold}', group=CFG.wandb_run_name) as run:
            gc_collect()
            train_loop(run, fold)

try:
    with open(f"{output_dir}/finished","w") as f:
        f.write("finish")
except:
    LOGGER.info("write finished fail.")


# %% [markdown]
# ## 6. Cross-validation

# %%
def gen_predictions(models, df_train, folds):
    df_train_predictions = []
    for model, fold in zip(models, folds):
        ds_valid = BreastCancerDataSet_16bit(df_train.query('split == @fold'), images_dir, CFG.target, get_transforms_16bit('valid', CFG.image_size, CFG.normalize_mean, CFG.normalize_std))
        valid_dataloader  = DataLoader(ds_valid, batch_size=CFG.valid_batch_size, shuffle=False, num_workers=CFG.num_workers, pin_memory=False)
        (pf1, thres), pred_cancer = valid_one_epoch(model, valid_dataloader)[:2]
        LOGGER.info(f'Eval fold:{fold} pF1:{pf1:.03f} thres:{thres:.02f}')
        
        df_pred = pd.DataFrame(data=pred_cancer, columns=['cancer_pred_proba'])
        df_pred['cancer_pred'] = df_pred["cancer_pred_proba"] > thres
        

        df = pd.concat(
            [df_train.query('split == @fold').reset_index(drop=True), df_pred],
            axis=1
        ).sort_values(['patient_id', 'image_id'])
        df_train_predictions.append(df)
    df_train_predictions = pd.concat(df_train_predictions)
    return df_train_predictions


if CV:
    models = [load_model(best_dict[fold])[0] for fold in CFG.folds]
    df_pred = gen_predictions(models, df_train, CFG.folds)
    df_pred.to_csv(f'{output_dir}/train_predictions.csv', index=False)
    # display(df_pred.head())

    LOGGER.info(f'F1 CV score (multiple thresholds): {sklearn.metrics.f1_score(df_pred["cancer"], df_pred["cancer_pred"])}')    

    df_pred_all = df_pred.groupby(['patient_id', 'laterality']).agg(
        cancer_max=('cancer_pred_proba', 'max'), cancer_mean=('cancer_pred_proba', 'mean'), cancer=('cancer', 'max')
    )
    LOGGER.info(f'ALL pF1 CV score. Mean , single threshold: {optimal_f1(df_pred_all["cancer"].values, df_pred_all["cancer_mean"].values)}', )
    LOGGER.info(f'ALL pF1 CV score. Max  , single threshold: {optimal_f1(df_pred_all["cancer"].values, df_pred_all["cancer_max"].values)}', )

    df_pred1 = df_pred[df_pred["site_id"]==1].groupby(['patient_id', 'laterality']).agg(
        cancer_max=('cancer_pred_proba', 'max'), cancer_mean=('cancer_pred_proba', 'mean'), cancer=('cancer', 'max')
    )
    LOGGER.info(f'SITE1 pF1 CV score. Mean , single threshold: {optimal_f1(df_pred1["cancer"].values, df_pred1["cancer_mean"].values)}', )
    LOGGER.info(f'SITE1 pF1 CV score. Max  , single threshold: {optimal_f1(df_pred1["cancer"].values, df_pred1["cancer_max"].values)}', )

    df_pred2 = df_pred[df_pred["site_id"]==2].groupby(['patient_id', 'laterality']).agg(
        cancer_max=('cancer_pred_proba', 'max'), cancer_mean=('cancer_pred_proba', 'mean'), cancer=('cancer', 'max')
    )
    LOGGER.info(f'SITE2 pF1 CV score. Mean , single threshold: {optimal_f1(df_pred2["cancer"].values, df_pred2["cancer_mean"].values)}', )
    LOGGER.info(f'SITE2 pF1 CV score. Max  , single threshold: {optimal_f1(df_pred2["cancer"].values, df_pred2["cancer_max"].values)}', )


