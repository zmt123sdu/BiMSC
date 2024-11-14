#!/usr/A/anaconda3/envs/py_3.8 python
# -*- coding: UTF-8 -*-
'''
@Project ：BiRC-DeepSC-EUR-8 
@File    ：utils_for_DeepSC.py
@Author  ：Mingtong Zhang
@Date    ：2024/2/14 15:24 
'''
import torch
from utils.utils_general import Channels, SNR_to_noise
import numpy as np


def create_key_masks(src, tgt, padding_idx, device):
    src_key_mask = (src == padding_idx).float().unsqueeze(1)  # [batch, 1, seq_len]
    tgt_key_mask = (tgt == padding_idx).float().unsqueeze(1)  # [batch, 1, seq_len]
    look_ahead_mask = torch.triu(torch.full((tgt.size(1), tgt.size(1)), float('1')), diagonal=1)

    return src_key_mask.to(device), tgt_key_mask.to(device), look_ahead_mask.to(device)


def PowerNormalize(x, key_mask, power_tx=1):
    # Note: if P = 1, the symbol power is 2
    # If you want to set the average power as 1, please change P as P=1/np.sqrt(2)

    x_square = torch.mul(x, x)
    real_symbol_num = torch.sum(key_mask, dim=1, keepdim=True) * x_square.size(-1)
    power = (torch.sum(x_square, dim=(-2, -1), keepdim=True) / real_symbol_num).sqrt()

    x = np.sqrt(power_tx) * torch.div(x, power)

    return x


def train_step(model, sent, pad, end, snr, opt, loss_function, channel, device):
    # 训练步骤开始
    model.train()
    opt.zero_grad()

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
    # 功率归一化
    tx_sig = PowerNormalize(jsc_enc_output, key_mask)

    channels = Channels()
    noise_std = SNR_to_noise(snr)
    if channel == 'AWGN':
        rx_sig = channels.AWGN(tx_sig, noise_std, device=device)
    elif channel == 'Rayleigh':
        rx_sig, _ = channels.Rayleigh(tx_sig, noise_std, device=device)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    # 屏蔽padding位置信道的影响
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


def eval_step(model, sent, pad, end, snr, loss_function, channel, device):
    # 训练步骤开始
    model.eval()

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
    # 功率归一化
    tx_sig = PowerNormalize(jsc_enc_output, key_mask)

    channels = Channels()
    noise_std = SNR_to_noise(snr)
    if channel == 'AWGN':
        rx_sig = channels.AWGN(tx_sig, noise_std, device=device)
    elif channel == 'Rayleigh':
        rx_sig, _ = channels.Rayleigh(tx_sig, noise_std, device=device)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    # 屏蔽padding位置信道的影响
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


def greedy_decode(model, sent, pad, end, snr, len_max, start_symbol, channel, device):
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
    # 功率归一化
    tx_sig = PowerNormalize(jsc_enc_output, key_mask)

    channels = Channels()
    noise_std = SNR_to_noise(snr)
    if channel == 'AWGN':
        rx_sig = channels.AWGN(tx_sig, noise_std, device=device)
    elif channel == 'Rayleigh':
        rx_sig, _ = channels.Rayleigh(tx_sig, noise_std, device=device)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

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

        dec_output = model.semantic_decoder(outputs, memory, look_ahead_mask=look_ahead_mask, tgt_key_mask=tgt_key_mask,
                                            memory_key_mask=src_key_mask)
        pred = model.dense(dec_output)
        prob = pred[:, -1:, :]  # (batch_size, 1, vocab_size)
        _, next_word = torch.max(prob, dim=-1)

        outputs = torch.cat([outputs, next_word], dim=1)

    return outputs
