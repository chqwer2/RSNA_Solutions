from utils.data_utils import *

# 0 ---> background
# 1 ---> C1
# 2 ---> C2
# ...
# 8 ---> T1 - T12

# ------------------ Helper --------------------

datadir = '../kingston'
seg_paths = glob(f"{datadir}/segmentations/*")
os.makedirs(f"{datadir}/seg_25d_image", exist_ok=True)
os.makedirs(f"{datadir}/seg_25d_mask", exist_ok=True)



seg_df = pd.DataFrame({'path': seg_paths})
seg_df['StudyInstanceUID'] = seg_df['path'].apply(lambda x:x.split('/')[-1][:-4])
seg_df = seg_df[['StudyInstanceUID', 'path']]

print('seg_df shape:', seg_df.shape)

study_uid_list = seg_df["StudyInstanceUID"].tolist()
dataframe_list = []

dataframe_list, study_uid_list = process_seg(datadir, study_uid_list, dataframe_list)


seg_25d_df = pd.DataFrame(dataframe_list,
                          columns=["id", "StudyInstanceUID",
                                   "slice_num", "image_path", "mask_path"])

set_folds(datadir, seg_25d_df)

