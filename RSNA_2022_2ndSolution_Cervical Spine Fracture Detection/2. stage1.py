

from options.CFG import CFG_stage1 as CFG
from utils.utils import *
from models.funcs import *

# ------------------ Profile ------------------

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

datadir = '../kingston'
libdir = '.'
outputdir = '.'
otherdir = '.'


package_paths = [f'{libdir}pytorch-image-models-master']
if CFG.device == 'TPU':
    # !pip     install - q     pytorch - ignite
    import ignite.distributed as idist

elif CFG.device == 'GPU':
    from torch.cuda.amp import autocast, GradScaler


# ------------------- Load --------------------

train_df = pd.read_csv(f'{datadir}/seg_25d.csv')
print('train_df shape:', train_df.shape)
train_df.head(3)


if CFG.device == 'TPU':
    import os
    VERSION = "1.7"
    CP_V = "36" if ENV == "colab" else "37"
    wheel = f"torch_xla-{VERSION}-cp{CP_V}-cp{CP_V}m-linux_x86_64.whl"
    url = f"https://storage.googleapis.com/tpu-pytorch/wheels/{wheel}"
    # !pip3 -q install cloud-tpu-client==0.10 $url
    os.system('export XLA_USE_BF16=1')
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
    CFG.lr = CFG.lr * CFG.nprocs
    CFG.train_bs = CFG.train_bs // CFG.nprocs
    device = xm.xla_device()

elif CFG.device == "GPU":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



seed_everything(CFG.seed)

LOGGER = init_logger(outputdir+f'/train{CFG.suffix}.log')

if CFG.device=='TPU' and CFG.nprocs==8:
    loginfo = xm.master_print
    cusprint = xm.master_print
else:
    loginfo = LOGGER.info
    cusprint = print



def main():
    for fold in range(CFG.fold_num):
        if fold in CFG.fold_list:
            # train_loop(train_df, fold)
            train_loop(train_df, CFG, fold, outputdir, device, LOGGER)


if __name__ == '__main__':
    print(CFG.suffix)
    if CFG.device == 'TPU':
        def _mp_fn(rank, flags):
            torch.set_default_tensor_type('torch.FloatTensor')
            a = main()
        FLAGS = {}
        xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=CFG.nprocs, start_method='fork')

    elif CFG.device == 'GPU':
        main()

