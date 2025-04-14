import os, sys, shutil, torch, traceback
from feature_extractor import cnhubert
from time import time as ttime

inp_wav_dir = os.environ.get("inp_wav_dir")
inp_text = os.environ.get("inp_text")
exp_name = os.environ.get("exp_name")
i_part = os.environ.get("i_part")
all_parts = os.environ.get("all_parts")
if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
opt_dir = os.environ.get("opt_dir")
now_dir = os.getcwd()
sys.path.append(now_dir)
is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()
device = "cuda" if torch.cuda.is_available() else "cpu"
