



from options.CFG_pseudo import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
datadir = '../kingston'
libdir = '.'
outputdir = '.'
otherdir = '.'

CFG.weight_path = f"{outputdir}/efficientnet-b0_109_fold0_epoch13.pth"

if CFG.device == 'TPU':
    # !pip install -q     pytorch-ignite
    import ignite.distributed as idist

elif CFG.device == 'GPU':
    from torch.cuda.amp import autocast, GradScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


seed_everything(CFG.seed)

# ------------------ Profile ------------------
study_train_df = pd.read_csv(f'{datadir}/train.csv')
print('train_df shape:', study_train_df.shape)
study_train_df.head(3)


seg_paths = glob(f"{datadir}/segmentations/*")
seg_gt_list = [path.split('/')[-1][:-4] for path in seg_paths]

study_train_df = study_train_df[~study_train_df["StudyInstanceUID"].isin(seg_gt_list)]
study_train_df.shape

train_slice_list = []
for file_name in tqdm(study_train_df["StudyInstanceUID"].values):
    train_image_path = glob(f"{datadir}/train_images/{file_name}/*")
    train_image_path = sorted(train_image_path, key=lambda x:int(x.split("/")[-1].replace(".dcm","")))
    for path_idx in range(len(train_image_path)):
        path1 = "nofile" if path_idx-1 < 0 else train_image_path[path_idx-1].replace(f"{datadir}/", "")
        path2 = train_image_path[path_idx].replace(f"{datadir}/", "")
        path3 = "nofile" if path_idx+1 >= len(train_image_path) else train_image_path[path_idx+1].replace(f"{datadir}/", "")
        slice_num = int(path2.split("/")[-1].replace(".dcm",""))
        train_slice_list.append([f"{file_name}_{slice_num}", file_name, slice_num, path1, path2, path3])

train_df = pd.DataFrame(train_slice_list, columns=["id", "StudyInstanceUID", "slice_num", "path1", "path2", "path3"])
train_df = train_df.sort_values(['StudyInstanceUID', 'slice_num'], ascending = [True, True]).reset_index(drop=True)
train_df.to_csv(f'{datadir}/train_slice_list.csv', index=False)

from models.model import *
from inference.pesudo import *


slice_class_list = []
voxel_crop_list = []
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

    if mask.shape[0] != 512 or mask.shape[1] != 512:
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
    mask = mask.transpose(2, 0, 1)
    assert mask.shape[1] == mask.shape[2] == 512
    mask = torch.from_numpy(mask).to(device, dtype=torch.uint8)
    croped_voxel, croped_voxel_mask = crop_voxel(mask, file_name)


# --------------- post-progress -----------------



import pandas as pd
import numpy as np
from tqdm import tqdm
datadir = '../kingston'


voxel_crop_df = pd.read_csv(f"{datadir}/voxel_crop.csv")
# voxel_crop_df = pd.DataFrame(voxel_crop_list, columns=["StudyInstanceUID", "before_image_size", "x0", "x1", "y0", "y1", "z0", "z1"]).sort_values(by=["StudyInstanceUID"])
# voxel_crop_df.to_csv(f"{datadir}/voxel_crop.csv", index=False)
voxel_crop_df # 每个study的整体crop坐标


slice_class_df = pd.read_csv(f"{datadir}/slice_class.csv")
# slice_class_df = pd.DataFrame(slice_class_list, columns=["StudyInstanceUID", "new_slice_num", "old_slice_num", "vertebra_class"]).sort_values(by=["StudyInstanceUID", "new_slice_num"])
# slice_class_df.to_csv(f"{datadir}/slice_class.csv", index=False)
slice_class_df # 每张slice的所属vertebra_class(preds)

# study_id_list = []
# slice_num_list = []
# for file_name in tqdm(voxel_crop_df["StudyInstanceUID"].values, total=len(voxel_crop_df)):
#     train_image_path = glob(f"{datadir}/train_images/{file_name}/*")
#     train_image_path = sorted(train_image_path, key=lambda x:int(x.split("/")[-1].replace(".dcm","")))
#     slice_cnt = len(train_image_path)

#     study_id_list.extend([file_name]*slice_cnt)
#     slice_num_list.extend([int(x.split("/")[-1].replace(".dcm","")) for x in train_image_path])

# all_slice_df = pd.DataFrame({"StudyInstanceUID":study_id_list, "slice_num":slice_num_list})
# all_slice_df.to_csv(f"{datadir}/all_slice_df.csv", index=False)
all_slice_df = pd.read_csv(f"{datadir}/all_slice_df.csv")
print(all_slice_df.shape)
all_slice_df.head(3)


new_df = []
for idx, study_id, _, x0, x1, _, _, _, _, in tqdm(voxel_crop_df.itertuples(), total=len(voxel_crop_df)):
    one_study = all_slice_df[all_slice_df["StudyInstanceUID"] == study_id].reset_index(drop=True)
    new_df.append(one_study[x0:x1])
new_df = pd.concat(new_df, axis=0).reset_index(drop=True)
new_df # 所有包含vertebra的slice

new_df = new_df.merge(voxel_crop_df, on="StudyInstanceUID", how="left") # merge study_crop_df
display(new_df) # 合并了study的crop信息
assert len(slice_class_df) == len(new_df)


new_slice_df = pd.concat([new_df, slice_class_df[["new_slice_num", "vertebra_class"]]], axis=1)
new_slice_df # 合并 class

tr_df = pd.read_csv(f"{datadir}/train.csv")
new_slice_df1 = new_slice_df.merge(tr_df, on="StudyInstanceUID", how="left")
new_slice_df1 # 合并 train.csv

new_slice_df1.to_csv(f"{datadir}/train_slice.csv", index=False)


sample_num = 24
vertebrae_df_list = []
for study_id in tqdm(np.unique(new_slice_df1["StudyInstanceUID"])):
    one_study = new_slice_df1[new_slice_df1["StudyInstanceUID"] == study_id].reset_index(drop=True)
    for cid in range(1, 8):
        one_study_cid = one_study[one_study["vertebra_class"] == cid].reset_index(drop=True)
        if len(one_study_cid) >= sample_num:
            sample_index = np.linspace(0, len(one_study_cid)-1, sample_num, dtype=int)
            one_study_cid = one_study_cid.iloc[sample_index].reset_index(drop=True)
        if len(one_study_cid) < 1:
            continue
        slice_num_list = one_study_cid["slice_num"].values.tolist()
        arow = one_study_cid.iloc[0]
        vertebrae_df_list.append([f"{study_id}_{cid}", study_id, cid, slice_num_list, arow["before_image_size"], \
            arow["x0"], arow["x1"], arow["y0"], arow["y1"], arow["z0"], arow["z1"], arow[f"C{cid}"]])

vertebrae_df = pd.read_pickle(f"{datadir}/vertebrae_df.pkl")


# Study Level
sample_num = 90
study_df_list = []
for study_id in tqdm(np.unique(new_slice_df1["StudyInstanceUID"])):
    one_study = new_slice_df1[new_slice_df1["StudyInstanceUID"] == study_id].reset_index(drop=True)
    if len(one_study) >= sample_num:
        sample_index = np.linspace(0, len(one_study)-1, sample_num, dtype=int)
        one_study = one_study.iloc[sample_index].reset_index(drop=True)
    slice_num_list = one_study["slice_num"].values.tolist()
    arow = one_study.iloc[0]
    study_df_list.append([study_id, slice_num_list, arow["before_image_size"], arow["x0"], arow["x1"], arow["y0"], arow["y1"], arow["z0"], arow["z1"], arow["patient_overall"]])


study_df = pd.DataFrame(study_df_list, columns=["StudyInstanceUID", "slice_num_list", "before_image_size", "x0", "x1", "y0", "y1", "z0", "z1", "label"])
study_df.to_pickle(f"{datadir}/study_df_{sample_num}.pkl")






