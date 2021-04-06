#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from .embedding import RelativePositionalEncoding, PositionalEncoding
from .encoder_layers import TransformerEncoderLayer
from .relative_attention import MultiHeadedAttention
from .modules import PositionwiseFeedForward


class TransformerEncoder(torch.nn.Module):
    """
    Transformer encoder module.

    :param int idim: input dim
    :param int attention_dim: dimention of attention
    :param int attention_heads: the number of heads of multi head attention
    :param int linear_units: the number of units of position-wise feed forward
    :param int num_blocks: the number of encoder blocks
    :param float dropout_rate: dropout rate
    :param float positional_dropout_rate: dropout rate after adding positional encoding
    :param float attention_dropout_rate: dropout rate in attention
    :param bool relative_pos_emb: whether to use relative position embedding
    :param bool absolute_pos_emb: whether to use absolute position embedding
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied. i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    """

    def __init__(self, idim,
                 attention_dim=256,
                 attention_heads=4,
                 linear_units=2048,
                 num_blocks=16,
                 dropout_rate=0.1,
                 positional_dropout_rate=0.1,
                 attention_dropout_rate=0.0,
                 ffn_bias=True,
                 ffn_activation_function='relu',
                 relative_pos_emb=True,
                 absolute_pos_emb=False,
                 normalize_before=True,
                 concat_after=False,
                 relative_v=False):
        """Construct an Encoder object."""
        super(TransformerEncoder, self).__init__()

        self.embed = torch.nn.Sequential(
            torch.nn.Linear(idim, attention_dim),
            torch.nn.LayerNorm(attention_dim),
            torch.nn.Dropout(dropout_rate),
            torch.nn.ReLU(),
        )

        if relative_pos_emb:
            print('w/ relative position embedding')
            self.rel_pos_emb = RelativePositionalEncoding(attention_dim // attention_heads, 1000, relative_v)
        else:
            print('w/o relative position embedding')
            self.rel_pos_emb = None

        if absolute_pos_emb:
            print('w/ absolute position embedding')
            self.abs_pos_emb = PositionalEncoding(attention_dim, 0.0)
        else:
            print('w/o absolute position embedding')
            self.abs_pos_emb = None

        self.dropout_layer = torch.nn.Dropout(p=positional_dropout_rate)
        self.normalize_before = normalize_before
        if self.normalize_before:
            self.after_norm = torch.nn.LayerNorm(attention_dim)

        self.encoders = torch.nn.Sequential(*[TransformerEncoderLayer(
                attention_dim,
                MultiHeadedAttention(attention_heads, attention_dim, attention_dropout_rate),
                PositionwiseFeedForward(attention_dim,
                                        linear_units,
                                        dropout_rate,
                                        bias=ffn_bias,
                                        activation_function=ffn_activation_function),
                dropout_rate,
                normalize_before,
                concat_after,
                relative_v,
                attention_heads
            ) for _ in range(num_blocks)])

    def forward(self, xs, masks=None, output_for_kd=None):
        """
        Args:
            xs (Tensor): N x T x F
            masks (Tensor or None): N x T x T
        Return:
            xs (Tensor): N x T x F
            masks (Tensor or None): N x T x T
        """
        xs = self.embed(xs)

        if self.rel_pos_emb is not None:
            x_len = xs.shape[1]
            pos_seq = torch.arange(0, x_len).long().to(xs.device)
            pos_seq = pos_seq[:, None] - pos_seq[None, :]
            pos_k, pos_v = self.rel_pos_emb(pos_seq)
        else:
            pos_k, pos_v = None, None

        if self.abs_pos_emb is not None:
            xs = self.abs_pos_emb(xs)

        xs = self.dropout_layer(xs)

        kd_values = {"output": {}}
        if output_for_kd and -1 in output_for_kd["output"]:
            kd_values["output"][-1] = xs
        for i, layer in enumerate(self.encoders):
            xs = layer(xs, pos_k, pos_v, masks)
            if output_for_kd and i in output_for_kd["output"]:
                kd_values["output"][i] = xs[0]
            xs = xs[0]

        if self.normalize_before:
            xs = self.after_norm(xs)

        return xs, kd_values
