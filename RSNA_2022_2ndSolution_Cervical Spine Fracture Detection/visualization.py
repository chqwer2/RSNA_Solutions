







# Stage1.ipynb
example_name = "1.2.826.0.1.3680043.780_147"
image_example = np.load(f"{datadir}/seg_25d_image/{example_name}.npy").transpose(2,0,1)
mask_example = np.load(f"{datadir}/seg_25d_mask/{example_name}.npy").transpose(2,0,1)
plt.figure(figsize=(30, 20))
plt.subplot(1, 3, 1); plt.imshow(image_example[1]); plt.axis('OFF');
plt.subplot(1, 3, 2); plt.imshow(mask_example[1]); plt.axis('OFF');
plt.subplot(1, 3, 3); plt.imshow(image_example[1]); plt.imshow(mask_example[1],alpha=0.5); plt.axis('OFF');
# plt.colorbar()
plt.tight_layout()
plt.show()




# 图像示例
from pylab import rcParams
dataset_show = TrainDataset(
    train_df,
    get_transforms("light_train") # None, get_transforms("train")
    )
rcParams['figure.figsize'] = 30,20
for i in range(2):
    f, axarr = plt.subplots(1,3)
    idx = np.random.randint(0, len(dataset_show))
    img, mask, raw_mask = dataset_show[idx]
    # axarr[p].imshow(img) # transform=None
    axarr[0].imshow(img[1]); plt.axis('OFF');
    axarr[1].imshow(raw_mask[1]/255, alpha=0.5); plt.axis('OFF');
    axarr[2].imshow(img[1]); axarr[2].imshow(raw_mask[1]/255,alpha=0.5); plt.axis('OFF');



# Pesudo
from pylab import rcParams
dataset_show = TrainDataset(
    train_df,
    get_transforms("valid") # None, get_transforms("train")
    )
rcParams['figure.figsize'] = 30,20
for i in range(2):
    f, axarr = plt.subplots(1,3)
    idx = np.random.randint(0, len(dataset_show))
    img, file_name, n_slice= dataset_show[idx]
    # axarr[p].imshow(img) # transform=None
    axarr[0].imshow(img[0]); plt.axis('OFF');
    axarr[1].imshow(img[1]); plt.axis('OFF');
    axarr[2].imshow(img[2]); plt.axis('OFF');


