#!/usr/A/anaconda3/envs/py_3.8 python
# -*- coding: UTF-8 -*-
'''
@Project ：BiRC-DeepSC-EUR-8 
@File    ：BiMSC.py
@Author  ：Mingtong Zhang
@Date    ：2024/9/25 16:42 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.DeepSC import Encoder, Decoder, JSC_Encoder, JSC_Decoder


class FANet(nn.Module):
    "Core encoder is a stack of num_layers layers"

    def __init__(self, num_input_parameter, max_channel_dim, mlp_ratio):
        super(FANet, self).__init__()

        self.max_channel_dim = max_channel_dim
        self.linear1 = nn.Linear(num_input_parameter, mlp_ratio * max_channel_dim)

        self.linear2 = nn.Linear(mlp_ratio * max_channel_dim, 2 * mlp_ratio * max_channel_dim)
        self.linear3 = nn.Linear(2 * mlp_ratio * max_channel_dim, mlp_ratio * max_channel_dim)

        self.linear4 = nn.Linear(mlp_ratio * max_channel_dim, 2 * mlp_ratio * max_channel_dim)
        self.linear5 = nn.Linear(2 * mlp_ratio * max_channel_dim, mlp_ratio * max_channel_dim)

        self.linear6 = nn.Linear(mlp_ratio * max_channel_dim, max_channel_dim)
        self.upper_tri_matrix = torch.triu(torch.ones((max_channel_dim, max_channel_dim)))

    def forward(self, x, rate, noise_std, len_sent, src_key_mask, device, threshold):
        key_mask = 1 - src_key_mask.transpose(-2, -1)
        if len(noise_std.size()) == 1:
            noise_std = torch.ones_like(rate) * noise_std.item()
        rate = rate / (self.max_channel_dim / 2)
        # 计算每个特征的均值和标准差
        fwm = torch.sum(x * key_mask, dim=1) / len_sent
        fwm = fwm.unsqueeze(1)
        fwstd = torch.sum((x - fwm) * (x - fwm) * key_mask, dim=1) / len_sent
        fwm = fwm.squeeze(1)

        x = torch.cat((rate, noise_std, fwm, fwstd), dim=1)

        x1 = self.linear1(x)

        x = F.relu(self.linear2(x1))
        x2 = F.relu(self.linear3(x) + x1)

        x = F.relu(self.linear4(x2))
        x = F.relu(self.linear5(x) + x2)

        soft_index = F.softmax(self.linear6(x), dim=-1)
        alpha = F.linear(soft_index, self.upper_tri_matrix.to(device))
        alpha = alpha - threshold
        hard_mask = torch.sign(torch.relu(alpha)) - alpha.detach().clone() + alpha

        return hard_mask


class BiMSC(nn.Module):
    def __init__(self, num_layers, src_vocab_size, tgt_vocab_size, model_dim, channel_dim, num_heads, dim_feedforward, dropout=0.1, mlp_ratio=2):
        super(BiMSC, self).__init__()

        self.fa_net = FANet(num_input_parameter=2 + 2 * channel_dim, max_channel_dim=channel_dim, mlp_ratio=mlp_ratio)

        self.semantic_encoder = Encoder(num_layers, src_vocab_size, model_dim, num_heads, dim_feedforward, dropout)

        self.jsc_encoder = JSC_Encoder(in_features_dim=model_dim, intermediate_dim=2 * model_dim, out_features_dim=channel_dim)

        self.jsc_decoder = JSC_Decoder(in_features_dim=channel_dim, intermediate_dim=2 * model_dim, out_features_dim=model_dim)

        self.semantic_decoder = Decoder(num_layers, tgt_vocab_size, model_dim, num_heads, dim_feedforward, dropout)

        self.dense = nn.Linear(model_dim, tgt_vocab_size)
