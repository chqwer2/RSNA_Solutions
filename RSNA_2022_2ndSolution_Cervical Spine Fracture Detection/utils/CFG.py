
train_bs_ = 16 # train_batch_size
valid_bs_ = 128 # valid_batch_size
num_workers_ = 5


libdir = '.'
outputdir = '.'
otherdir = '.'



class CFG:
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


class pseudo_CFG:
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
    weight_path = f"{outputdir}/efficientnet-b0_109_fold0_epoch13.pth"


class stage2_CFG:
    seed = 42
    device = 'GPU'
    nprocs = 1  # [1, 8]
    num_workers = 5
    train_bs = 2
    valid_bs = 4
    fold_num = 5

    target_cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    num_classes = 7

    accum_iter = 4
    max_grad_norm = 1000
    print_freq = 100
    normalize_mean = [0.4824, 0.4824, 0.4824]  # [0.485, 0.456, 0.406] [0.4824, 0.4824, 0.4824]
    normalize_std = [0.22, 0.22, 0.22]  # [0.229, 0.224, 0.225] [0.22, 0.22, 0.22]

    suffix = "406"
    fold_list = [0, 1, 2, 3, 4]
    epochs = 20
    model_arch = "resnest50d"  # tf_efficientnetv2_s, resnest50d
    img_size = 400
    optimizer = "AdamW"
    scheduler = "CosineAnnealingLR"
    loss_fn = "BCEWithLogitsLoss"
    scheduler_warmup = "GradualWarmupSchedulerV3"

    warmup_epo = 1
    warmup_factor = 10
    T_max = epochs - warmup_epo - 2

    seq_len = 24
    lr = 5e-5
    min_lr = 1e-7
    weight_decay = 0
    dropout = 0.1

    gpu_parallel = False
    n_early_stopping = 5
    debug = False
    multihead = False