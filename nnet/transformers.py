#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch import nn
from .encoders import TransformerEncoder

default_encoder_conf = {
    "transformer": {
            "attention_dim": 256,
            "attention_heads": 4,
            "linear_units": 2048,
            "num_blocks": 16,
            "dropout_rate": 0.1,
            "positional_dropout_rate": 0.1,
            "attention_dropout_rate": 0.0,
            "relative_pos_emb": True,
            "absolute_pos_emb": False,
            "normalize_before": True,
            "concat_after": False,
            "relative_v": False
        }
}


class Transformer(nn.Module):
    """
    Transformers models:
        Transformers encoders + masks estimator
    """
    def __init__(self,
                 in_features=257,
                 num_bins=257,
                 num_spks=2,
                 num_nois=1,
                 architecture="transformer",
                 transformer_conf=default_encoder_conf["transformer"]):
        super(Transformer, self).__init__()

        # Transformers
        support_architecture = {
            "transformer": TransformerEncoder,
        }
        self.transformer = support_architecture[architecture](in_features, **transformer_conf)

        # linear
        self.num_bins = num_bins
        self.num_spks = num_spks
        self.num_nois = num_nois
        self.linear = nn.Linear(transformer_conf["attention_dim"], num_bins * (num_spks + num_nois))

        # non-linear
        self.mask_non_linear = torch.sigmoid

    def forward(self, f, output_for_kd=None):
        # Transformers encoders
        f, kd_values = self.transformer(f, output_for_kd=output_for_kd)

        # estimate masks
        masks = self.linear(f)
        masks = self.mask_non_linear(masks)

        # N x T x F => N x F x T
        masks = masks.transpose(1, 2)
        masks = torch.chunk(masks, self.num_spks + self.num_nois, 1)

        if output_for_kd:
            return masks, kd_values
        return masks


class FreqTransformer(Transformer):
    """
    Frequency domain Transformers
    """
    def __init__(self,
                 stats_file=None,
                 in_features=257,
                 num_bins=257,
                 num_spks=2,
                 num_nois=1,
                 enh_transform=None,
                 training_mode="freq",
                 architecture="transformer",
                 transformer_conf=default_encoder_conf):
        super(FreqTransformer, self).__init__(in_features=in_features,
                                              num_bins=num_bins,
                                              num_spks=num_spks,
                                              num_nois=num_nois,
                                              architecture=architecture,
                                              transformer_conf=transformer_conf)

        # input normalization layer
        if stats_file is not None:
            stats = np.load(stats_file)
            self.input_bias = torch.from_numpy(np.tile(np.expand_dims(-stats['mean'].astype(np.float32), axis=0), (1, 1, 1)))
            self.input_scale = torch.from_numpy(np.tile(np.expand_dims(1 / np.sqrt(stats['variance'].astype(np.float32)), axis=0), (1, 1, 1)))
            self.input_bias = nn.Parameter(self.input_bias, requires_grad=False)
            self.input_scale = nn.Parameter(self.input_scale, requires_grad=False)
        else:
            self.input_bias = torch.zeros(1,1,in_features)
            self.input_scale = torch.ones(1,1,in_features)
            self.input_bias = nn.Parameter(self.input_bias, requires_grad=False)
            self.input_scale = nn.Parameter(self.input_scale, requires_grad=False)

        if enh_transform is None:
            raise RuntimeError("enh_transform can not be None")
        self.enh_transform = enh_transform
        self.mode = training_mode

    def _forward(self, mix, mode, output_for_kd=None):
        # mix_feat: N x T x F
        # mix_stft: N x (C) x F x T
        mix_feat, mix_stft, _ = self.enh_transform(mix, None)
        # global feature normalization
        mix_feat = mix_feat + self.input_bias
        mix_feat = mix_feat * self.input_scale
        if mix_stft and mix_stft.dim() == 4:
            # N x F x T
            mix_stft = mix_stft[:, 0]

        # [N x F x T] x S
        masks = super().forward(mix_feat, output_for_kd=output_for_kd)

        # output masks
        if mode == "freq":
            return masks
        else:
            masks = masks[0] if output_for_kd else masks
            decoder = self.enh_transform.inverse_stft
            if self.num_spks == 1:
                enh_stft = mix_stft * masks
                enh = decoder((enh_stft.real, enh_stft.imag), input="complex")
            else:
                enh_stft = [mix_stft * m for m in masks]
                enh = [
                    decoder((s.real, s.imag), input="complex")
                    for s in enh_stft
                ]
            return enh

    def infer(self, mix, mode="time"):
        """
        Args:
            mix (Tensor): (C) x S
        Return:
            sep [Tensor, ...]: S or
            masks [Tensor, ...]: F x T
        """
        with torch.no_grad():
            mix = mix[None, ...]
            spk = self._forward(mix, mode)
            return [s[0] for s in spk]

    def forward(self, mix, output_for_kd=None):
        """
        Args:
            mix (Tensor): N x (C) x S
        Return:
            masks [Tensor, ...]: N x F x T or
            spks [Tensor, ...]: N x S
        """
        return self._forward(mix, self.mode, output_for_kd=output_for_kd)