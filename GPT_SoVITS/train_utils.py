import warnings, utils, logging, os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import logging, traceback
from random import randint
from module import commons

from module.data_utils import (
    TextAudioSpeakerLoaderV3 as TextAudioSpeakerLoader,
    TextAudioSpeakerCollateV3 as TextAudioSpeakerCollate,
    DistributedBucketSampler,
)
from module.models import (
    SynthesizerTrnV3 as SynthesizerTrn,
    MultiPeriodDiscriminator,
)
from module.losses import generator_loss, discriminator_loss, feature_loss, kl_loss
from module.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from process_ckpt import savee

warnings.filterwarnings("ignore")
hps = utils.get_hparams(stage=2)
os.environ["CUDA_VISIBLE_DEVICES"] = hps.train.gpu_numbers.replace("-", ",")
logging.getLogger("matplotlib").setLevel(logging.INFO)
logging.getLogger("h5py").setLevel(logging.INFO)
logging.getLogger("numba").setLevel(logging.INFO)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
###反正A100fp32更快，那试试tf32吧
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")  # 最低精度但最快（也就快一丁点），对于结果造成不了影响
# from config import pretrained_s2G,pretrained_s2D
global_step = 0

device = "cpu"  # cuda以外的设备，等mps优化后加入



