# -*- coding: utf-8 -*-

import sys, os

inp_text = os.environ.get("inp_text")
inp_wav_dir = os.environ.get("inp_wav_dir")
exp_name = os.environ.get("exp_name")
i_part = os.environ.get("i_part")
all_parts = os.environ.get("all_parts")
if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
from feature_extractor import cnhubert

opt_dir = os.environ.get("opt_dir")
cnhubert.cnhubert_base_path = os.environ.get("cnhubert_base_dir")
import torch, librosa

import numpy as np
from scipy.io import wavfile
from tools.my_utils import clean_path, i18n, ffmpeg, load_audio
import shutil, numba
from basic_util import *
from scipy import signal
from functools import lru_cache

io_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=estimate_safe_workers())
compute_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=min(4, estimate_safe_workers()))  # Limited for GPU compute
process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=estimate_safe_workers())
_model_instance = None
_device = None
_is_half = None


async def my_save(fea, path):
    """Optimized save function with better error handling"""
    import tempfile, uuid

    loop = asyncio.get_event_loop()
    dir_path = os.path.dirname(path)

    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    temp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(temp_dir, f"{int(ttime())}_{uuid.uuid4().hex[:8]}.pth")

    try:
        # Combine the save and move operations into a single function to reduce overhead
        def _save_and_move():
            torch.save(fea, tmp_path)
            shutil.move(tmp_path, path)

        await loop.run_in_executor(compute_thread_pool, _save_and_move)
    except Exception as e:
        print(f"保存中出错: {str(e)}")
        if os.path.exists(tmp_path):
            await loop.run_in_executor(compute_thread_pool, functools.partial(os.remove, tmp_path))
        raise


def get_hubert_model(device, is_half):
    """Singleton pattern to reuse HubertModel instance"""
    global _model_instance, _device, _is_half

    if _model_instance is None or device != _device or is_half != _is_half:
        _model_instance = cnhubert.get_model()
        _model_instance = _model_instance.half().to(device) if is_half else _model_instance.to(device)
        _model_instance.eval()
        _device = device
        _is_half = is_half

    return _model_instance


def fast_resample(audio, orig_sr, target_sr):
    """
    Fast audio resampling using scipy.signal.resample_poly
    """
    if orig_sr == target_sr:
        return audio

    gcd = np.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd

    return signal.resample_poly(audio, up, down, axis=0)


@functools.lru_cache(maxsize=8)
def scaling_factors(maxx, alpha):
    """Precompute scaling factors to avoid redundant calculations."""
    return maxx * alpha * 32768, (1 - alpha) * 32768, maxx * alpha * 1145.14, (1 - alpha) * 1145.14


def do_audio(audio, max_val, factor1, factor2, factor3, factor4):
    """Process audio with optimized computations."""
    normalized = audio / max_val
    audio32 = (normalized * factor1) + (factor2 * audio)
    audio32b = (normalized * factor3) + (factor4 * audio)
    return audio32, audio32b


def p_audio(wav_name, wav_path, device, is_half, hubert_dir, wav32dir, f1, f2, f3, f4):
    """Optimized audio processing with model reuse"""
    hubert_path = f"{hubert_dir}/{wav_name}.pt"
    if os.path.exists(hubert_path):
        return None

    tmp_audio = load_audio(wav_path, 32000)
    tmp_max = np.abs(tmp_audio).max()

    if tmp_max > 2.2:
        print(f"{wav_name}-filtered,{tmp_max}")
        return None

    tmp_audio32, tmp_audio32b = do_audio(tmp_audio, tmp_max, f1, f2, f3, f4)
    tmp_audio = fast_resample(tmp_audio32b, 32000, 16000)

    # Convert to tensor in one step
    tensor_wav16 = torch.from_numpy(tmp_audio).to(device)
    if is_half:
        tensor_wav16 = tensor_wav16.half()
    tensor_wav16 = tensor_wav16.unsqueeze(0)

    # Get the shared model instance
    model = get_hubert_model(device, is_half)

    with torch.no_grad():
        ssl = model.model(tensor_wav16)["last_hidden_state"].transpose(1, 2).cpu()

    if np.isnan(ssl.detach().numpy()).sum() != 0:
        print(f"nan filtered:{wav_name}")
        return wav_name, wav_path

    try:
        asyncio.run(my_save(ssl, hubert_path))
    except Exception as e:
        print(f"Error saving outputs for {wav_name}: {str(e)}")
        return wav_name, wav_path

    return None


@perf_check
def main():
    global _model_instance

    is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()
    hubert_dir = f"{opt_dir}/4-cnhubert"
    wav32dir = f"{opt_dir}/5-wav32k"

    os.makedirs(opt_dir, exist_ok=True)
    os.makedirs(hubert_dir, exist_ok=True)
    os.makedirs(wav32dir, exist_ok=True)

    maxx = 0.95
    alpha = 0.5
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Initialize model once at start
    get_hubert_model(device, is_half)

    with open(inp_text, "r", encoding="utf8") as f:
        lines = f.read().strip("\n").split("\n")

    # Precompute scaling factors once
    factors = scaling_factors(maxx, alpha)

    batch_size = 16
    selected_lines = lines[int(i_part)::int(all_parts)]

    for i in range(0, len(selected_lines), batch_size):
        batch_lines = selected_lines[i:i + batch_size]
        for line in batch_lines:
            try:
                wav_name, spk_name, language, text = line.split("|")
                wav_name = clean_path(wav_name)
                if inp_wav_dir != "" and inp_wav_dir is not None:
                    wav_name = os.path.basename(wav_name)
                    wav_path = f"{inp_wav_dir}/{wav_name}"
                else:
                    wav_path = wav_name
                    wav_name = os.path.basename(wav_name)

                p_audio(wav_name, wav_path, device, is_half, hubert_dir, wav32dir, *factors)
            except:
                print(f'{line}', traceback.format_exc())

    # Clean up model instance at the end
    _model_instance = None


if __name__ == "__main__":
    main()
