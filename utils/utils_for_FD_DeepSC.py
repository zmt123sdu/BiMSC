#!/usr/A/anaconda3/envs/py_3.8 python
# -*- coding: UTF-8 -*-
'''
@Project ：BiRC-DeepSC-EUR-8 
@File    ：utils_for_FD_DeepSC.py
@Author  ：Mingtong Zhang
@Date    ：2024/2/20 20:14 
'''
import torch
from utils.utils_general import Channels, SNR_to_noise
import numpy as np


def create_key_masks(src, tgt, padding_idx, device):
    src_key_mask = (src == padding_idx).float().unsqueeze(1)  # [batch, 1, seq_len]
    tgt_key_mask = (tgt == padding_idx).float().unsqueeze(1)  # [batch, 1, seq_len]
    look_ahead_mask = torch.triu(torch.full((tgt.size(1), tgt.size(1)), float('1')), diagonal=1)

    return src_key_mask.to(device), tgt_key_mask.to(device), look_ahead_mask.to(device)


def PowerNormalize(x, key_mask, mask=None, power_tx=1):
    x_square = torch.mul(x, x)

    if mask is not None:
        combine_mask = mask * key_mask
        real_symbol_num = torch.sum(combine_mask, dim=(-2, -1), keepdim=True)
        power = (torch.sum(x_square, dim=(-2, -1), keepdim=True) / real_symbol_num).sqrt()
    else:
        real_symbol_num = torch.sum(key_mask, dim=1, keepdim=True) * x_square.size(-1)
        power = (torch.sum(x_square, dim=(-2, -1), keepdim=True) / real_symbol_num).sqrt()

    x = np.sqrt(power_tx) * torch.div(x, power)

    return x


def train_step(model, sent, pad, end, snr, opt, loss_function, channel, rate_min, rate_max, device):
    # 训练步骤开始
    model.train()
    opt.zero_grad()
    noise_std = SNR_to_noise(snr)

    condition_tensor = (sent == end).clone().detach()
    sent_no_end = torch.where(condition_tensor, pad, sent)
    src = sent_no_end[:, 1: -1]
    tgt_inp = sent_no_end[:, :-1]
    tgt_real = sent[:, 1:]

    # 产生掩码
    src_key_mask, tgt_key_mask, look_ahead_mask = create_key_masks(src, tgt_inp, pad, device=device)
    key_mask = 1 - src_key_mask.transpose(-2, -1)

    # 语义编码
    enc_output = model.semantic_encoder(src, src_key_mask=src_key_mask)
    # 联合信源信道编码
    jsc_enc_output = model.jsc_encoder(enc_output)
    jsc_enc_output = jsc_enc_output * key_mask

    # 采样出实际传输速率
    batch_dim = src.size(0)
    channel_dim = jsc_enc_output.size(-1)
    rate = torch.randint(rate_min, rate_max + 1, (batch_dim, 1)).to(device)

    # 进行特征维度掩码
    feature_dim = torch.round(2 * rate).int()
    fd_mask = torch.ones(batch_dim, 1, channel_dim).to(device)
    for bs in range(batch_dim):
        fd_mask[bs, :, feature_dim[bs, 0]:] = 0

    jsc_enc_output = jsc_enc_output * fd_mask

    # 功率归一化
    tx_sig = PowerNormalize(jsc_enc_output, key_mask, fd_mask)

    channels = Channels()
    if channel == 'AWGN':
        rx_sig = channels.AWGN(tx_sig, noise_std, device=device)
    elif channel == 'Rayleigh':
        rx_sig, _ = channels.Rayleigh(tx_sig, noise_std, device=device)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    # 移除掩码部分的噪声影响
    rx_sig = rx_sig * fd_mask
    # 屏蔽padding的影响
    rx_sig = rx_sig * key_mask
    # 联合信源信道译码
    jsc_dec_output = model.jsc_decoder(rx_sig)
    # 语义译码
    dec_output = model.semantic_decoder(tgt_inp, jsc_dec_output, look_ahead_mask=look_ahead_mask, tgt_key_mask=tgt_key_mask, memory_key_mask=src_key_mask)
    pred = model.dense(dec_output)
    # 计算Loss
    ntokens = pred.size(-1)
    loss = loss_function(pred.reshape(-1, ntokens), tgt_real.reshape(-1))

    loss.backward()
    opt.step()

    return loss.item()


def eval_step(model, sent, pad, end, snr, loss_function, channel, rate_min, rate_max, device):
    # 训练步骤开始
    model.eval()
    noise_std = SNR_to_noise(snr)

    condition_tensor = (sent == end).clone().detach()
    sent_no_end = torch.where(condition_tensor, pad, sent)
    src = sent_no_end[:, 1: -1]
    tgt_inp = sent_no_end[:, :-1]
    tgt_real = sent[:, 1:]

    # 产生掩码
    src_key_mask, tgt_key_mask, look_ahead_mask = create_key_masks(src, tgt_inp, pad, device=device)
    key_mask = 1 - src_key_mask.transpose(-2, -1)

    # 语义编码
    enc_output = model.semantic_encoder(src, src_key_mask=src_key_mask)
    # 联合信源信道编码
    jsc_enc_output = model.jsc_encoder(enc_output)
    jsc_enc_output = jsc_enc_output * key_mask

    # 采样出实际传输速率
    batch_dim = src.size(0)
    channel_dim = jsc_enc_output.size(-1)
    rate = torch.randint(rate_min, rate_max + 1, (batch_dim, 1)).to(device)

    # 进行特征维度掩码
    feature_dim = torch.round(2 * rate).int()
    fd_mask = torch.ones(batch_dim, 1, channel_dim).to(device)
    for bs in range(batch_dim):
        fd_mask[bs, :, feature_dim[bs, 0]:] = 0

    jsc_enc_output = jsc_enc_output * fd_mask

    # 功率归一化
    tx_sig = PowerNormalize(jsc_enc_output, key_mask, fd_mask)

    channels = Channels()
    if channel == 'AWGN':
        rx_sig = channels.AWGN(tx_sig, noise_std, device=device)
    elif channel == 'Rayleigh':
        rx_sig, _ = channels.Rayleigh(tx_sig, noise_std, device=device)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    # 移除掩码部分的噪声影响
    rx_sig = rx_sig * fd_mask
    # 屏蔽padding的影响
    rx_sig = rx_sig * key_mask
    # 联合信源信道译码
    jsc_dec_output = model.jsc_decoder(rx_sig)
    # 语义译码
    dec_output = model.semantic_decoder(tgt_inp, jsc_dec_output, look_ahead_mask=look_ahead_mask, tgt_key_mask=tgt_key_mask, memory_key_mask=src_key_mask)
    pred = model.dense(dec_output)
    # 计算Loss
    ntokens = pred.size(-1)
    loss = loss_function(pred.reshape(-1, ntokens), tgt_real.reshape(-1))

    return loss.item()


def greedy_decode(model, sent, pad, end, snr, len_max, start_symbol, channel, rate, device):
    model.eval()

    condition_tensor = (sent == end).clone().detach()
    sent_no_end = torch.where(condition_tensor, pad, sent)
    src = sent_no_end[:, 1: -1]

    # 产生掩码
    src_key_mask, _, _ = create_key_masks(src, src, pad, device=device)
    key_mask = 1 - src_key_mask.transpose(-2, -1)

    # 语义编码
    enc_output = model.semantic_encoder(src, src_key_mask=src_key_mask)
    # 联合信源信道编码
    jsc_enc_output = model.jsc_encoder(enc_output)
    jsc_enc_output = jsc_enc_output * key_mask

    # 进行特征维度掩码
    batch_dim = src.size(0)
    channel_dim = jsc_enc_output.size(-1)
    feature_dim = 2 * rate
    fd_mask = torch.ones(batch_dim, 1, channel_dim).to(device)
    fd_mask[:, :, feature_dim:] = 0
    jsc_enc_output = jsc_enc_output * fd_mask

    # 功率归一化
    tx_sig = PowerNormalize(jsc_enc_output, key_mask, fd_mask)

    channels = Channels()
    noise_std = SNR_to_noise(snr)
    if channel == 'AWGN':
        rx_sig = channels.AWGN(tx_sig, noise_std, device=device)
    elif channel == 'Rayleigh':
        rx_sig, _ = channels.Rayleigh(tx_sig, noise_std, device=device)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    # 移除掩码部分的噪声影响
    rx_sig = rx_sig * fd_mask
    # 屏蔽padding的影响
    rx_sig = rx_sig * key_mask
    # 联合信源信道译码
    memory = model.jsc_decoder(rx_sig)

    # 自回归贪婪译码
    outputs = torch.ones(src.size(0), 1).fill_(start_symbol).type_as(src.data)
    for i in range(len_max + 1):
        # create the decode mask
        tgt_key_mask = (outputs == pad).float().unsqueeze(1).to(device)  # [batch, 1, seq_len]
        look_ahead_mask = torch.triu(torch.full((outputs.size(-1), outputs.size(-1)), float('1')), diagonal=1).to(device)

        # decode the received signal
        dec_output = model.semantic_decoder(outputs, memory, look_ahead_mask=look_ahead_mask, tgt_key_mask=tgt_key_mask,
                                            memory_key_mask=src_key_mask)
        pred = model.dense(dec_output)

        # predict the word
        prob = pred[:, -1:, :]  # (batch_size, 1, vocab_size)

        # return the max-prob index
        _, next_word = torch.max(prob, dim=-1)

        outputs = torch.cat([outputs, next_word], dim=1)

    return outputs
