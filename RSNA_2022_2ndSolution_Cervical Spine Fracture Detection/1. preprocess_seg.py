from utils.data_utils import *
from utils.seg_utils1 import *
from CFG import CFG

# 0 ---> background
# 1 ---> C1
# 2 ---> C2
# ...
# 8 ---> T1 - T12


datadir = '../kingston'

seed_everything(CFG.seed)
LOGGER = init_logger(outputdir+f'/train{CFG.suffix}.log')

if CFG.device=='TPU' and CFG.nprocs==8:
    loginfo = xm.master_print
    cusprint = xm.master_print
else:
    loginfo = LOGGER.info
    cusprint = print

def get_result(result_df):
    preds = result_df[[f'pred_{c}' for c in CFG.target_cols]].values
    labels = result_df[CFG.target_cols].values
    score, scores = get_score(labels, preds)
    LOGGER.info(f'Score: {score:<.4f}  Scores: {np.round(scores, decimals=4)}')



# ------------------ Helper --------------------

# Store segmentation paths in a dataframe
seg_paths = glob(f"{datadir}/segmentations/*")
seg_df = pd.DataFrame({'path': seg_paths})
seg_df['StudyInstanceUID'] = seg_df['path'].apply(lambda x:x.split('/')[-1][:-4])
seg_df = seg_df[['StudyInstanceUID','path']]
print('seg_df shape:', seg_df.shape)
seg_df.head(3)


study_uid_list = seg_df["StudyInstanceUID"].tolist()
dataframe_list = []
os.makedirs(f"{datadir}/seg_25d_image", exist_ok=True)
os.makedirs(f"{datadir}/seg_25d_mask", exist_ok=True)

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




seg_25d_df = pd.DataFrame(dataframe_list, columns=["id", "StudyInstanceUID", "slice_num", "image_path", "mask_path"])
seg_25d_df["fold"] = -1

gkf = GroupKFold(n_splits=5)
for idx, (train_index, test_index) in enumerate(gkf.split(X=seg_25d_df, groups=seg_25d_df['StudyInstanceUID'].values)):
    seg_25d_df.loc[test_index, 'fold'] = idx

    for i in range(5):
        study_num = len(np.unique(seg_25d_df[seg_25d_df["fold"] == i]["StudyInstanceUID"]))
        print(f"fold{i} num: {study_num}")

    seg_25d_df.to_csv(f"{datadir}/seg_25d.csv", index=False)

