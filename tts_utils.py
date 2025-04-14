import re, torch, yaml, librosa, threading
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer
from basic_util import *
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from feature_extractor.cnhubert import CNHubert
from module.models import SynthesizerTrn
from text.LangSegmenter import LangSegmenter
from text import chinese
from typing import Dict, List, Tuple, Union
from text.cleaner import clean_text
from text import cleaned_text_to_sequence
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm
from tools.my_utils import load_audio
from module.mel_processing import spectrogram_torch
from tools.i18n.i18n import I18nAuto, scan_language_list
import numpy as np
from inf_utils import merge_short_text_in_array, filter_text
from seg_utils import *
import time
from time import time as ttime

sys.path.append(now_dir)
language = os.environ.get("language", "Auto")
language = sys.argv[-1] if sys.argv[-1] in scan_language_list() else language
i18n = I18nAuto(language=language)



