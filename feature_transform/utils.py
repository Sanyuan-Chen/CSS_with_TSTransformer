import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as tf
import librosa.filters as filters
from scipy.fftpack import dct

MATH_PI = math.pi
EPSILON = np.finfo(np.float32).eps
MAX_INT16 = np.iinfo(np.int16).max


def init_window(wnd, frame_len):
    """
    Return window coefficient
    """
    def sqrthann(frame_len):
        return th.hann_window(frame_len)**0.5

    if wnd not in ["bartlett", "hann", "hamm", "blackman", "rect", "sqrthann"]:
        raise RuntimeError(f"Unknown window type: {wnd}")

    wnd_tpl = {
        "sqrthann": sqrthann,
        "hann": th.hann_window,
        "hamm": th.hamming_window,
        "blackman": th.blackman_window,
        "bartlett": th.bartlett_window,
        "rect": th.ones
    }
    c = wnd_tpl[wnd](frame_len)
    return c


def init_kernel(frame_len,
                frame_hop,
                window,
                round_pow_of_two=True,
                normalized=False,
                inverse=False):
    """
    Return STFT kernels
    """
    # FFT points
    B = 2**math.ceil(math.log2(frame_len)) if round_pow_of_two else frame_len
    if normalized:
        # make K^H * K = I
        S = B**0.5
    else:
        S = 1
    I = th.stack([th.eye(B), th.zeros(B, B)], dim=-1)
    # W x B x 2
    K = th.fft(I / S, 1)[:frame_len]
    if inverse and not normalized:
        # to make K^H * K = I
        K = K / B
    # 2 x B x W
    K = th.transpose(K, 0, 2) * window
    # 2B x 1 x W
    K = th.reshape(K, (B * 2, 1, frame_len))
    return K


def init_melfilter(frame_len,
                   round_pow_of_two=True,
                   num_bins=None,
                   sr=16000,
                   num_mels=80,
                   fmin=0.0,
                   fmax=None):
    """
    Return mel-filters
    """
    # FFT points
    if num_bins is None:
        N = 2**math.ceil(
            math.log2(frame_len)) if round_pow_of_two else frame_len
    else:
        N = (num_bins - 1) * 2
    # fmin & fmax
    fmax = sr // 2 if fmax is None else min(fmax, sr // 2)
    # mel-matrix
    mel = filters.mel(sr, N, n_mels=num_mels, fmax=fmax, fmin=fmin, htk=True)
    # num_mels x (N // 2 + 1)
    return th.tensor(mel, dtype=th.float32)


def init_dct(num_ceps=13, num_mels=40):
    """
    Return DCT matrix
    """
    dct_mat = dct(np.eye(num_mels), norm="ortho")[:num_ceps]
    # num_ceps x num_mels
    return th.tensor(dct_mat, dtype=th.float32)


def _forward_stft(wav,
                  kernel,
                  output="polar",
                  frame_hop=256,
                  onesided=False,
                  center=False):
    """
    STFT inner function
    Args:
        wav (Tensor), N x (C) x S
        kernel (Tensor), STFT transform kernels, from init_kernel(...)
        output (str), output format:
            polar: return (magnitude, phase) pair
            complex: return (real, imag) pair
            real: return [real; imag] Tensor
        frame_hop: frame hop size in number samples
        onesided: return half FFT bins
        center: if true, we assumed to have centered frames
    Return:
        transform (Tensor or [Tensor, Tensor]), STFT transform results
    """
    wav_dim = wav.dim()
    if output not in ["polar", "complex", "real"]:
        raise ValueError(f"Unknown output format: {output}")
    if wav_dim not in [2, 3]:
        raise RuntimeError(f"STFT expect 2D/3D tensor, but got {wav_dim:d}D")
    # if N x S, reshape N x 1 x S
    # else: reshape NC x 1 x S
    N, S = wav.shape[0], wav.shape[-1]
    wav = wav.view(-1, 1, S)
    # NC x 1 x S+2P
    if center:
        ps = (frame_hop, frame_hop, 0, 0, 0, 0)
        wav = tf.pad(wav, ps, mode="constant", value=0)
    # STFT
    packed = tf.conv1d(wav, kernel, stride=frame_hop, padding=0)
    # NC x 2B x T => N x C x 2B x T
    if wav_dim == 3:
        packed = packed.view(N, -1, packed.shape[-2], packed.shape[-1])
    # N x (C) x B x T
    real, imag = th.chunk(packed, 2, dim=-2)
    # N x (C) x B/2+1 x T
    if onesided:
        num_bins = kernel.shape[0] // 4 + 1
        real = real[..., :num_bins, :]
        imag = imag[..., :num_bins, :]
    if output == "complex":
        return (real, imag)
    elif output == "real":
        return th.stack([real, imag], dim=-1)
    else:
        mag = (real**2 + imag**2 + EPSILON)**0.5
        pha = th.atan2(imag, real)
        return (mag, pha)


def _inverse_stft(transform,
                  kernel,
                  window,
                  input="polar",
                  frame_hop=256,
                  onesided=False,
                  center=False):
    """
    iSTFT inner function
    Args:
        transform (Tensor or [Tensor, Tensor]), STFT transform results        
        kernel (Tensor), STFT transform kernels, from init_kernel(...)
        input (str), input format:
            polar: return (magnitude, phase) pair
            complex: return (real, imag) pair
            real: return [real; imag] Tensor
        frame_hop: frame hop size in number samples
        onesided: return half FFT bins
        center: used in _forward_stft
    Return:
        wav (Tensor), N x S
    """
    if input not in ["polar", "complex", "real"]:
        raise ValueError(f"Unknown output format: {input}")

    if input == "real":
        real, imag = transform[..., 0], transform[..., 1]
    elif input == "polar":
        real = transform[0] * th.cos(transform[1])
        imag = transform[0] * th.sin(transform[1])
    else:
        real, imag = transform

    # (N) x F x T
    imag_dim = imag.dim()
    if imag_dim not in [2, 3]:
        raise RuntimeError(f"Expect 2D/3D tensor, but got {imag_dim}D")

    # if F x T, reshape 1 x F x T
    if imag_dim == 2:
        real = th.unsqueeze(real, 0)
        imag = th.unsqueeze(imag, 0)

    if onesided:
        # [self.num_bins - 2, ..., 1]
        reverse = range(kernel.shape[0] // 4 - 1, 0, -1)
        # extend matrix: N x B x T
        real = th.cat([real, real[:, reverse]], 1)
        imag = th.cat([imag, -imag[:, reverse]], 1)
    # pack: N x 2B x T
    packed = th.cat([real, imag], dim=1)
    # N x 1 x T
    s = tf.conv_transpose1d(packed, kernel, stride=frame_hop, padding=0)
    # normalized audio samples
    # refer: https://github.com/pytorch/audio/blob/2ebbbf511fb1e6c47b59fd32ad7e66023fa0dff1/torchaudio/functional.py#L171
    # 1 x W x T
    win = th.repeat_interleave(window[None, ..., None],
                               packed.shape[-1],
                               dim=-1)
    # W x 1 x W
    I = th.eye(window.shape[0], device=win.device)[:, None]
    # 1 x 1 x T
    norm = tf.conv_transpose1d(win**2, I, stride=frame_hop, padding=0)
    if center:
        s = s[..., frame_hop:-frame_hop]
        norm = norm[..., frame_hop:-frame_hop]
    s = s / (norm + EPSILON)
    # N x S
    s = s.squeeze(1)
    return s


def forward_stft(wav,
                 frame_len,
                 frame_hop,
                 output="complex",
                 window="sqrthann",
                 round_pow_of_two=True,
                 normalized=False,
                 onesided=True,
                 center=False):
    """
    STFT function implementation, equals to STFT layer
    """
    w = init_window(window, frame_len)
    K = init_kernel(frame_len,
                    frame_hop,
                    w,
                    round_pow_of_two=round_pow_of_two,
                    normalized=normalized,
                    inverse=False)
    return _forward_stft(wav,
                         K.to(wav.device),
                         output=output,
                         frame_hop=frame_hop,
                         onesided=onesided,
                         center=center)


def inverse_stft(transform,
                 frame_len,
                 frame_hop,
                 input="complex",
                 window="sqrthann",
                 round_pow_of_two=True,
                 normalized=False,
                 onesided=True,
                 center=False):
    """
    iSTFT function implementation, equals to iSTFT layer
    """
    if isinstance(transform, th.Tensor):
        device = transform.device
    else:
        device = transform[0].device
    w = init_window(window, frame_len)
    K = init_kernel(frame_len,
                    frame_hop,
                    w,
                    round_pow_of_two=round_pow_of_two,
                    normalized=normalized,
                    inverse=True)
    return _inverse_stft(transform,
                         K.to(device),
                         w.to(device),
                         input=input,
                         frame_hop=frame_hop,
                         onesided=onesided,
                         center=center)


class STFTBase(nn.Module):
    """
    Base layer for (i)STFT
    """
    def __init__(self,
                 frame_len,
                 frame_hop,
                 window="sqrthann",
                 round_pow_of_two=True,
                 normalized=False,
                 onesided=True,
                 inverse=False,
                 center=False):
        super(STFTBase, self).__init__()
        w = init_window(window, frame_len)
        K = init_kernel(frame_len,
                        frame_hop,
                        w,
                        round_pow_of_two=round_pow_of_two,
                        normalized=normalized,
                        inverse=inverse)
        self.K = nn.Parameter(K, requires_grad=False)
        self.w = nn.Parameter(w, requires_grad=False)
        self.frame_len = frame_len
        self.frame_hop = frame_hop
        self.onesided = onesided
        self.center = center
        self.num_bins = self.K.shape[0] // 4 + 1
        self.expr = (
            f"window={window}, stride={frame_hop}, onesided={onesided}, " +
            f"normalized={normalized}, center={self.center}, " +
            f"kernel_size={self.num_bins}x{self.K.shape[2]}")

    def num_frames(self, num_samples):
        if th.sum(num_samples <= self.frame_len):
            raise RuntimeError(f"Audio samples {num_samples.cpu()} less " +
                               f"than frame_len ({self.frame_len})")
        return (num_samples - self.frame_len) // self.frame_hop + 1

    def extra_repr(self):
        return self.expr


class STFT(STFTBase):
    """
    Short-time Fourier Transform as a Layer
    """
    def __init__(self, *args, **kwargs):
        super(STFT, self).__init__(*args, inverse=False, **kwargs)

    def forward(self, wav, output="polar"):
        """
        Accept (single or multiple channel) raw waveform and output magnitude and phase
        Args
            wav (Tensor) input signal, N x (C) x S
        Return
            transform (Tensor or [Tensor, Tensor]), N x (C) x F x T
        """
        return _forward_stft(wav,
                             self.K,
                             output=output,
                             frame_hop=self.frame_hop,
                             onesided=self.onesided,
                             center=self.center)


class iSTFT(STFTBase):
    """
    Inverse Short-time Fourier Transform as a Layer
    """
    def __init__(self, *args, **kwargs):
        super(iSTFT, self).__init__(*args, inverse=True, **kwargs)

    def forward(self, transform, input="polar"):
        """
        Accept phase & magnitude and output raw waveform
        Args
            transform (Tensor or [Tensor, Tensor]), STFT output
        Return
            s (Tensor), N x S
        """
        return _inverse_stft(transform,
                             self.K,
                             self.w,
                             input=input,
                             frame_hop=self.frame_hop,
                             onesided=self.onesided,
                             center=self.center)