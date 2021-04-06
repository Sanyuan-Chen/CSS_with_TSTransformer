#!/usr/bin/env python

"""
Feature transform Separation
"""

import math, random
import torch as th
import torch.nn as nn
from torch_complex.tensor import ComplexTensor
from feature_transform.utils import STFT, iSTFT, EPSILON, MATH_PI


class AbsTransform(nn.Module):
    """
    Absolute transform
    """
    def __init__(self, eps=EPSILON):
        super(AbsTransform, self).__init__()
        self.eps = eps

    def extra_repr(self):
        return f"eps={self.eps:.3e}"

    def forward(self, x):
        """
        Args:
            x (Tensor or ComplexTensor): N x T x F
        Return:
            y (Tensor): N x T x F
        """
        if not isinstance(x, th.Tensor):
            x = x + self.eps
        return x.abs()


class LogTransform(nn.Module):
    """
    Transform linear domain to log domain
    """
    def __init__(self, eps=1e-5):
        super(LogTransform, self).__init__()
        self.eps = eps

    def dim_scale(self):
        return 1

    def extra_repr(self):
        return f"eps={self.eps:.3e}"

    def forward(self, x):
        """
        Args:
            x (Tensor): linear, N x (C) x T x F
        Return:
            y (Tensor): log features, N x (C) x T x F
        """
        x = th.clamp(x, min=self.eps)
        return th.log(x)


class CmvnTransform(nn.Module):
    """
    Utterance-level mean-variance normalization
    """
    def __init__(self, norm_mean=True, norm_var=True, gcmvn="", eps=1e-5):
        super(CmvnTransform, self).__init__()
        self.gmean, self.gstd = None, None
        if gcmvn:
            stats = th.load(gcmvn)
            mean, std = stats[0], stats[1]
            self.gmean = nn.Parameter(mean, requires_grad=False)
            self.gstd = nn.Parameter(std, requires_grad=False)
        self.norm_mean = norm_mean
        self.norm_var = norm_var
        self.gcmvn = gcmvn
        self.eps = eps

    def extra_repr(self):
        return f"norm_mean={self.norm_mean}, norm_var={self.norm_var}, " + \
            f"gcmvn_stats={self.gcmvn}, eps={self.eps:.3e}"

    def dim_scale(self):
        return 1

    def forward(self, x):
        """
        Args:
            x (Tensor): feature before normalization, N x (C) x T x F
        Return:
            y (Tensor): normalized feature, N x (C) x T x F
        """
        if not self.norm_mean and not self.norm_var:
            return x
        # over time axis
        m = th.mean(x, -2, keepdim=True) if self.gmean is None else self.gmean
        if self.gstd is None:
            ms = th.mean(x**2, -2, keepdim=True)
            s = (ms - m**2 + self.eps)**0.5
        else:
            s = self.gstd
        if self.norm_mean:
            x = x - m
        if self.norm_var:
            x = x / s
        return x


def random_mask(shape, max_steps=30, num_masks=2, order="freq", device="cpu"):
    """
    Generate random 0/1 masks
    Args:
        shape: (T, F)
    Return:
        masks (Tensor): 0,1 masks, T x F
    """
    if order not in ["time", "freq"]:
        raise RuntimeError(f"Unknown order: {order}")
    # shape: T x F
    masks = th.ones(shape, device=device)
    L = shape[1] if order == "freq" else shape[0]
    for _ in range(num_masks):
        dur = random.randint(1, max_steps - 1)
        if L - dur <= 0:
            continue
        beg = random.randint(0, L - dur - 1)
        if order == "freq":
            masks[:, beg:beg + dur] = 0
        else:
            masks[beg:beg + dur, :] = 0
    return masks


def tf_mask(batch,
            shape,
            max_bands=30,
            max_frame=40,
            num_freq_masks=2,
            num_time_masks=2,
            device="cpu"):
    """
    Return batch of TF-masks
    Args:
        batch: batch size, N
        shape: (T x F)
    Return:
        masks (Tensor): 0,1 masks, N x T x F
    """
    T, F = shape
    max_frame = min(max_frame, T // 2)
    max_bands = min(max_bands, F // 2)
    mask = []
    for _ in range(batch):
        fmask = random_mask(shape,
                            max_steps=max_bands,
                            num_masks=num_freq_masks,
                            order="freq",
                            device=device)
        tmask = random_mask(shape,
                            max_steps=max_frame,
                            num_masks=num_time_masks,
                            order="time",
                            device=device)
        mask.append(fmask * tmask)
    # N x T x F
    return th.stack(mask)


class SpecAugTransform(nn.Module):
    """
    Spectra data augmentation
    """
    def __init__(self,
                 p=0.5,
                 max_bands=30,
                 max_frame=40,
                 num_freq_masks=2,
                 num_time_masks=2):
        super(SpecAugTransform, self).__init__()
        self.fnum, self.tnum = num_freq_masks, num_time_masks
        self.F, self.T = max_bands, max_frame
        self.p = p

    def extra_repr(self):
        return f"max_bands={self.F}, max_frame={self.T}, p={self.p}, " \
                + f"num_freq_masks={self.fnum}, num_time_masks={self.tnum}"

    def forward(self, x):
        """
        Args:
            x (Tensor): original features, N x (C) x T x F
        Return:
            y (Tensor): augmented features
        """
        if self.training and th.rand(1).item() < self.p:
            if x.dim() == 4:
                N, _, T, F = x.shape
            else:
                N, T, F = x.shape
            # N x T x F
            mask = tf_mask(N, (T, F),
                           max_bands=self.F,
                           max_frame=self.T,
                           num_freq_masks=self.fnum,
                           num_time_masks=self.tnum,
                           device=x.device)
            if x.dim() == 4:
                # N x 1 x T x F
                mask = mask.unsqueeze(1)
            x = x * mask
        return x


class IpdTransform(nn.Module):
    """
    Compute inter-channel phase difference
    """
    def __init__(self, ipd_index="1,0", cos=True, sin=False, normalize="none"):
        super(IpdTransform, self).__init__()
        if normalize not in ["none", "v1", "v2", "v3"]:
            raise ValueError(f"Unknown IPD normalization: {normalize}")
        split_index = lambda sstr: [
            tuple(map(int, p.split(","))) for p in sstr.split(";")
        ]
        # ipd index
        pair = split_index(ipd_index)
        self.index_l = [t[0] for t in pair]
        self.index_r = [t[1] for t in pair]
        self.ipd_index = ipd_index
        self.cos = cos
        self.sin = sin
        self.num_pairs = len(pair) * 2 if cos and sin else len(pair)
        self.normalize = normalize

    def extra_repr(self):
        return (
            f"ipd_index={self.ipd_index}, cos={self.cos}, sin={self.sin}, " +
            f"normalize={self.normalize}")

    def forward(self, p):
        """
        Accept multi-channel phase and output inter-channel phase difference
        Args
            p (Tensor): phase matrix, N x C x F x T
        Return
            ipd (Tensor): IPD features,  N x MF x T
        """
        if p.dim() not in [3, 4]:
            raise RuntimeError(
                "{} expect 3/4D tensor, but got {:d} instead".format(
                    self.__class__.__name__, p.dim()))
        # C x F x T => 1 x C x F x T
        if p.dim() == 3:
            p = p.unsqueeze(0)
        N, _, _, T = p.shape
        pha_dif = p[:, self.index_l] - p[:, self.index_r]
        if self.normalize != "none":
            if self.normalize == "v3":
                pha_dif_mean = pha_dif.mean(-1, keepdim=True)
                pha_dif -= pha_dif_mean
            else:
                yr = th.cos(pha_dif)
                yi = th.sin(pha_dif)
                yrm = yr.mean(-1, keepdim=True)
                yim = yi.mean(-1, keepdim=True)
                if self.normalize == "v1":
                    pha_dif = th.atan2(yi - yim, yr - yrm)
                else:
                    pha_dif_mean = th.atan2(yim, yrm)
                    pha_dif -= pha_dif_mean
        if self.cos:
            # N x M x F x T
            ipd = th.cos(pha_dif)
            if self.sin:
                # N x M x 2F x T, along frequency axis
                ipd = th.cat([ipd, th.sin(pha_dif)], 2)
        else:
            ipd = th.fmod(pha_dif + math.pi, 2 * math.pi) - math.pi
            # ipd = th.where(ipd > MATH_PI, ipd - MATH_PI * 2, ipd)
            # ipd = th.where(ipd <= -MATH_PI, ipd + MATH_PI * 2, ipd)
        # N x MF x T
        ipd = ipd.view(N, -1, T)
        # N x MF x T
        return ipd


class DfTransform(nn.Module):
    """
    Compute angle/directional feature
        1) num_doas == 1: we known the DoA of the target speaker
        2) num_doas != 1: we do not have that prior, so we sampled #num_doas DoAs 
                          and compute on each directions    
    """
    def __init__(self,
                 geometric="princeton",
                 sr=16000,
                 velocity=343,
                 num_bins=257,
                 num_doas=1,
                 af_index="1,0;2,0;3,0;4,0;5,0;6,0"):
        super(DfTransform, self).__init__()
        if geometric not in ["princeton"]:
            raise RuntimeError(f"Unsupported array geometric: {geometric}")
        self.geometric = geometric
        self.sr = sr
        self.num_bins = num_bins
        self.num_doas = num_doas
        self.velocity = velocity
        split_index = lambda sstr: [
            tuple(map(int, p.split(","))) for p in sstr.split(";")
        ]
        # ipd index
        pair = split_index(af_index)
        self.index_l = [t[0] for t in pair]
        self.index_r = [t[1] for t in pair]
        self.af_index = af_index
        omega = th.tensor(
            [math.pi * sr * f / (num_bins - 1) for f in range(num_bins)])
        # 1 x F
        self.omega = nn.Parameter(omega[None, :], requires_grad=False)

    def _oracle_phase_delay(self, doa):
        """
        Compute oracle phase delay given DoA
        Args
            doa (Tensor): N
        Return
            phi (Tensor): N x (D) x C x F
        """
        device = doa.device
        if self.num_doas != 1:
            # doa is a unused, fake parameter
            N = doa.shape[0]
            # N x D
            doa = th.linspace(0, MATH_PI * 2, self.num_doas + 1,
                              device=device)[:-1].repeat(N, 1)
        # for princeton
        # M = 7, R = 0.0425, treat M_0 as (0, 0)
        #      *3    *2
        #
        #   *4    *0    *1
        #
        #      *5    *6
        if self.geometric == "princeton":
            R = 0.0425
            zero = th.zeros_like(doa)
            # N x 7 or N x D x 7
            tau = R * th.stack([
                zero, -th.cos(doa), -th.cos(MATH_PI / 3 - doa),
                -th.cos(2 * MATH_PI / 3 - doa),
                th.cos(doa),
                th.cos(MATH_PI / 3 - doa),
                th.cos(2 * MATH_PI / 3 - doa)
            ],
                               dim=-1) / self.velocity
            # (Nx7x1) x (1xF) => Nx7xF or (NxDx7x1) x (1xF) => NxDx7xF
            phi = th.matmul(tau.unsqueeze(-1), -self.omega)
            return phi
        else:
            return None

    def extra_repr(self):
        return (
            f"geometric={self.geometric}, af_index={self.af_index}, " +
            f"sr={self.sr}, num_bins={self.num_bins}, velocity={self.velocity}, "
            + f"known_doa={self.num_doas == 1}")

    def _compute_af(self, ipd, doa):
        """
        Compute angle feature
        Args
            ipd (Tensor): N x C x F x T
            doa (Tensor): DoA of the target speaker (if we known that), N 
                 or N x D (we do not known that, sampling D DoAs instead)
        Return
            af (Tensor): N x (D) x F x T
        """
        # N x C x F or N x D x C x F
        d = self._oracle_phase_delay(doa)
        d = d.unsqueeze(-1)
        if self.num_doas == 1:
            dif = d[:, self.index_l] - d[:, self.index_r]
            # N x C x F x T
            af = th.cos(ipd - dif)
            # on channel dimention (mean or sum)
            af = th.mean(af, dim=1)
        else:
            # N x D x C x F x 1
            dif = d[:, :, self.index_l] - d[:, :, self.index_r]
            # N x D x C x F x T
            af = th.cos(ipd.unsqueeze(1) - dif)
            # N x D x F x T
            af = th.mean(af, dim=2)
        return af

    def forward(self, p, doa):
        """
        Accept doa of the speaker & multi-channel phase, output angle feature
        Args
            doa (Tensor or list[Tensor]): DoA of target/each speaker, N or [N, ...]
            p (Tensor): phase matrix, N x C x F x T
        Return
            af (Tensor): angle feature, N x F* x T or N x D x F x T (known_doa=False)
        """
        if p.dim() not in [3, 4]:
            raise RuntimeError(
                "{} expect 3/4D tensor, but got {:d} instead".format(
                    self.__class__.__name__, p.dim()))
        # C x F x T => 1 x C x F x T
        if p.dim() == 3:
            p = p.unsqueeze(0)
        ipd = p[:, self.index_l] - p[:, self.index_r]

        if isinstance(doa, list):
            if self.num_doas != 1:
                raise RuntimeError("known_doa=False, no need to pass "
                                   "doa as a Sequence object")
            # [N x F x T or N x D x F x T, ...]
            af = [self._compute_af(ipd, spk_doa) for spk_doa in doa]
            # N x F x T => N x F* x T
            af = th.cat(af, 1)
        else:
            # N x F x T or N x D x F x T
            af = self._compute_af(ipd, doa)
        return af


class FixedBeamformer(nn.Module):
    """
    Fixed beamformer as a layer
    """
    def __init__(self,
                 num_beams,
                 num_channels,
                 num_bins,
                 weight=None,
                 requires_grad=False):
        super(FixedBeamformer, self).__init__()
        if weight:
            # (2, num_directions, num_channels, num_bins)
            w = th.load(weight)
            if w.shape[1] != num_beams:
                raise RuntimeError(f"Number of beam got from {w.shape[1]} " +
                                   f"don't match parameter {num_beams}")
            self.init_weight = weight
        else:
            self.init_weight = None
            w = th.zeros(2, num_beams, num_channels, num_bins)
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        # (num_directions, num_channels, num_bins, 1)
        self.real = nn.Parameter(w[0].unsqueeze(-1),
                                 requires_grad=requires_grad)
        self.imag = nn.Parameter(w[1].unsqueeze(-1),
                                 requires_grad=requires_grad)
        self.requires_grad = requires_grad

    def extra_repr(self):
        B, M, F, _ = self.real.shape
        return (f"num_beams={B}, num_channels={M}, " +
                f"num_bins={F}, init_weight={self.init_weight}, " +
                f"requires_grad={self.requires_grad}")

    def forward(self, x, beam=None, squeeze=False, trans=False, cplx=True):
        """
        Args:
            x (Complex Tensor): N x C x F x T
            beam (Tensor or None): N
        Return:
            1) (Tensor, Tensor): N x (B) x F x T
            2) (ComplexTensor): N x (B) x F x T
        """
        r, i = x.real, x.imag
        if r.dim() != i.dim() and r.dim() != 4:
            raise RuntimeError(
                f"FixBeamformer accept 4D tensor, got {r.dim()}")
        if self.real.shape[1] != r.shape[1]:
            raise RuntimeError(f"Number of channels mismatch: "
                               f"{r.shape[1]} vs {self.real.shape[1]}")
        if beam is None:
            # output all the beam
            br = th.sum(r.unsqueeze(1) * self.real, 2) + th.sum(
                i.unsqueeze(1) * self.imag, 2)
            bi = th.sum(i.unsqueeze(1) * self.real, 2) - th.sum(
                r.unsqueeze(1) * self.imag, 2)
        else:
            # output selected beam
            br = th.sum(r * self.real[beam], 1) + th.sum(
                i * self.imag[beam], 1)
            bi = th.sum(i * self.real[beam], 1) - th.sum(
                r * self.imag[beam], 1)
        if squeeze:
            br = br.squeeze()
            bi = bi.squeeze()
        if trans:
            br = br.transpose(-1, -2)
            bi = bi.transpose(-1, -2)
        if cplx:
            return ComplexTensor(br, bi)
        else:
            return br, bi


class FeatureTransform(nn.Module):
    """
    Feature transform for Enhancement/Separation tasks
    Spectrogram - LogTransform - CmvnTransform + IpdTransform
    NOTE: To using fixed beamformer or angle feature, please include it in the 
    network definition
    """
    def __init__(self,
                 feats="spectrogram-log-cmvn",
                 frame_len=512,
                 frame_hop=256,
                 window="sqrthann",
                 round_pow_of_two=True,
                 stft_normalized=False,
                 center=False,
                 sr=16000,
                 gcmvn="",
                 norm_mean=True,
                 norm_var=True,
                 aug_prob=0,
                 aug_max_bands=90,
                 aug_max_frame=40,
                 num_aug_bands=2,
                 num_aug_frame=2,
                 ipd_index="",
                 ipd_norm="none",
                 cos_ipd=True,
                 sin_ipd=False,
                 eps=EPSILON):
        super(FeatureTransform, self).__init__()
        self.frame_len = frame_len
        self.frame_hop = frame_hop
        self.stft_kwargs = {
            "window": window,
            "center": center,
            "normalized": stft_normalized,
            "round_pow_of_two": round_pow_of_two
        }
        # instance (i)STFT for use
        self.forward_stft = self.ctx(name="forward_stft")
        self.inverse_stft = self.ctx(name="inverse_stft")

        trans_tokens = feats.split("-") if feats else []
        transform = []
        feats_dim = 0
        feats_ipd = None
        for i, tok in enumerate(trans_tokens):
            if i == 0:
                if tok != "spectrogram" and tok != "ipd":
                    raise RuntimeError("Now only support spectrogram features")
                feats_dim = self.forward_stft.num_bins
            if tok == "spectrogram":
                transform.append(AbsTransform(eps=eps))
            elif tok == "log":
                transform.append(LogTransform(eps=eps))
            elif tok == "cmvn":
                transform.append(
                    CmvnTransform(norm_mean=norm_mean,
                                  norm_var=norm_var,
                                  gcmvn=gcmvn,
                                  eps=eps))
            elif tok == "ipd":
                feats_ipd = IpdTransform(ipd_index=ipd_index,
                                         cos=cos_ipd,
                                         sin=sin_ipd,
                                         normalize=ipd_norm)
                ipd_index = ipd_index.split(";")
                base = 0 if i == 0 else 1
                if cos_ipd and sin_ipd:
                    feats_dim *= (len(ipd_index) * 2 + base)
                else:
                    feats_dim *= (len(ipd_index) + base)
            else:
                raise RuntimeError(f"Unknown token {tok} in {feats}")
        if len(transform):
            self.mag_transform = nn.Sequential(*transform)
        else:
            self.mag_transform = None
        self.ipd_transform = feats_ipd
        if aug_prob > 0:
            self.aug_transform = SpecAugTransform(p=aug_prob,
                                                  max_bands=aug_max_bands,
                                                  max_frame=aug_max_frame,
                                                  num_freq_masks=num_aug_bands,
                                                  num_time_masks=num_aug_frame)
        else:
            self.aug_transform = None
        self.feats_dim = feats_dim

    def ctx(self, name="forward_stft"):
        """
        Return ctx(STFT/iSTFT) for task defined in src/aps/task
        """
        ctx = nn.ModuleDict({
            "forward_stft":
            STFT(self.frame_len, self.frame_hop, **self.stft_kwargs),
            "inverse_stft":
            iSTFT(self.frame_len, self.frame_hop, **self.stft_kwargs)
        })
        if name not in ctx:
            raise ValueError(f"Unknown task context: {name}")
        return ctx[name]

    def forward(self, wav_pad, wav_len, norm_obs=False):
        """
        Args:
            wav_pad (Tensor): raw waveform, N x C x S or N x S
            wav_len (Tensor or None): number samples in wav_pad, N or None
        Return:
            feats (Tensor): spatial + spectral features, N x T x ...
            cplx (ComplexTensor): STFT of reference channels, N x (C) x F x T
            feats_len (Tensor or None): number frames in each batch, N or None
        """
        # N x C x F x T
        mag, pha = self.forward_stft(wav_pad)
        multi_channel = mag.dim() == 4
        mag_ref = mag[:, 0] if multi_channel else mag

        # spectral (magnitude) transform
        if self.mag_transform:
            # N x T x F
            feats = mag_ref.transpose(-1, -2)
            # spectra features of CH0, N x T x F
            feats = self.mag_transform(feats)
            if self.aug_transform:
                # spectra augmentation if needed
                feats = self.aug_transform(feats)
        else:
            feats = None
            # spectra augmentation if needed
            if self.aug_transform:
                mag = self.aug_transform(mag)
        # complex spectrogram of CH 0~(C-1), N x C x F x T
        if norm_obs and multi_channel:
            mag_norm = th.norm(mag, p=2, dim=1, keepdim=True)
            mag = mag / th.clamp(mag_norm, min=EPSILON)
        cplx = ComplexTensor(mag * th.cos(pha), mag * th.sin(pha))
        # ipd transform
        if self.ipd_transform:
            # N x T x ...
            ipd = self.ipd_transform(pha)
            # N x ... x T
            ipd = ipd.transpose(1, 2)
            # N x T x ...
            if feats is not None:
                feats = th.cat([feats, ipd], -1)
            else:
                feats = ipd
        feats_len = self.forward_stft.num_frames(
            wav_len) if wav_len is not None else None
        return feats, cplx, feats_len