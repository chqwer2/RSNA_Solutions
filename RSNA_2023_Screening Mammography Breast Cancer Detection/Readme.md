# 8th place Solution

Before everything starts, I would like to say a big thank you to kaggle and the contest organizers for creating such an admirable contest.



## DataSet

We chose Nvidia Dali, which can decode dicom image files using the GPU, and generated uint16 bit png files. We did not use more complex algorithms for crop, but just `cv2.connectedComponentsWithStats`, which works very well and is fast.

Cross-validation strategy：`StratifiedGroupKFold` 5-Folds

negative samples strategy :  35%~50% of negative samples downsampled

Data augmentation strategies with different levels of LIGHT and HEAVY:

```python
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
```

Pseudo label: We simply used the Vindr data as an external dataset, using almost the same data processing approach.



## Models

Because of our training time, we chose the smallest version of the major models to try as much as possible, and ended up using three models Ensemble, which have a number of parameters between 15M and 21M, with broadly similar model training parameters, with some tuning of the parameters, which according to our experiments have a huge and sensitive impact on the learning rate.

- tf_efficientnetv2_s，lr: 1e-4
- convnext_nano， lr: 7e-6
- eca_nfnet_l0，lr: 3e-5

After Backbone, we choose `GeM Pooling`, p_trainable=True, and add `dropout` of fc layer.



## Training

- 3 Stage Training
  - 1. Training  with competition data
  - 2. Training  with  pseudo data
  - 3. Finetune with competition data
- Params：
  - AdamW，weight_decay = 0.01
  - Loss：BCEWithLogitsLoss
  - scheduler：OneCycleLR



# Inference

- Horizontal flip tta

- Binarization post-processing

  

## Doesn't work or doesn't do

1. RCNN/Yolo to crop
1. Larger sizes such as 2048
2. Focal Loss
4. More external Data
5. site1 and site2 Threshold
