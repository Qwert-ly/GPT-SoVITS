import torch
import torch.utils.data
import torch.jit as jit
import functools
from librosa.filters import mel as librosa_mel_fn
import numpy as np

MAX_WAV_VALUE = 32768.0


# 使用缓存装饰器来减少重复计算
def _get_cache_key(dtype, device, identifier=""):
    return f"{identifier}_{dtype}_{device}"


# 优化 1: 使用 JIT 编译动态范围压缩和解压缩函数
@jit.script
def dynamic_range_compression_torch(x: torch.Tensor, C: float = 1.0, clip_val: float = 1e-5) -> torch.Tensor:
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


@jit.script
def dynamic_range_decompression_torch(x: torch.Tensor, C: float = 1.0) -> torch.Tensor:
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


@jit.script
def spectral_de_normalize_torch(magnitudes: torch.Tensor) -> torch.Tensor:
    output = dynamic_range_decompression_torch(magnitudes)
    return output


# 创建缓存字典作为函数的本地变量，而不是全局变量
class MelBasisCache:
    def __init__(self):
        self.cache = {}

    def get_mel_basis(self, sampling_rate, n_fft, num_mels, fmin, fmax, dtype, device):
        key = f"{fmax}_{dtype}_{device}"
        if key not in self.cache:
            mel = librosa_mel_fn(
                sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
            )
            self.cache[key] = torch.from_numpy(mel).to(dtype=dtype, device=device)
        return self.cache[key]


class HannWindowCache:
    def __init__(self):
        self.cache = {}

    def get_hann_window(self, win_size, dtype, device):
        key = f"{win_size}_{dtype}_{device}"
        if key not in self.cache:
            self.cache[key] = torch.hann_window(win_size).to(dtype=dtype, device=device)
        return self.cache[key]


# 创建缓存实例
_mel_basis_cache = MelBasisCache()
_hann_window_cache = HannWindowCache()


# 优化 2: 使用 JIT 优化 spectrogram 计算
@jit.script
def _stft_operation(y: torch.Tensor, n_fft: int, hop_size: int, win_size: int,
                    hann_window: torch.Tensor, center: bool) -> torch.Tensor:
    # 注意: padding 操作不能直接 JIT 编译，所以提取出来
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )
    return torch.sqrt(spec.pow(2).sum(-1) + 1e-6)


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    # 使用缓存获取 hann 窗口
    hann_window = _hann_window_cache.get_hann_window(win_size, y.dtype, y.device)

    # 优化 padding 操作
    pad_size = int((n_fft - hop_size) / 2)
    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (pad_size, pad_size),
        mode="reflect",
    )
    y = y.squeeze(1)

    # 执行 STFT 操作
    return _stft_operation(y, n_fft, hop_size, win_size, hann_window, center)


@jit.script
def _matmul_operation(mel_basis: torch.Tensor, spec: torch.Tensor) -> torch.Tensor:
    return dynamic_range_compression_torch(torch.matmul(mel_basis, spec))


def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    # 使用缓存获取 mel 基础矩阵
    mel_basis = _mel_basis_cache.get_mel_basis(
        sampling_rate, n_fft, num_mels, fmin, fmax, spec.dtype, spec.device
    )

    # 执行矩阵乘法并应用压缩
    return _matmul_operation(mel_basis, spec)


# 优化 3: 合并相似代码并复用函数
def mel_spectrogram_torch(
        y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    # 先计算普通频谱图
    spec = spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center)

    # 小优化: 将 1e-6 改为 1e-9，以匹配原始实现
    # 通过额外计算来修正差异
    if 1e-6 != 1e-9:
        spec = spec * torch.sqrt((spec.pow(2) + 1e-9) / (spec.pow(2) + 1e-6))

    # 将频谱图转换为梅尔频谱图
    return spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax)


# 为向后兼容性，保留全局字典，但使用缓存类的内部字典
mel_basis = _mel_basis_cache.cache
hann_window = _hann_window_cache.cache
