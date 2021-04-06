#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch


class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward

    :param int idim: input dimenstion
    :param int hidden_units: number of hidden units
    :param float dropout_rate: dropout rate
    """

    def __init__(self, idim, hidden_units, dropout_rate, bias=True, activation_function='relu'):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units, bias=bias)
        if activation_function == 'relu':
            self.act = torch.nn.ReLU()
        else:
            raise ValueError
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.w_2 = torch.nn.Linear(hidden_units, idim, bias=bias)

    def forward(self, x):
        out = self.w_1(x)
        out = self.act(out)
        out = self.dropout(out)
        out = self.w_2(out)
        return out
