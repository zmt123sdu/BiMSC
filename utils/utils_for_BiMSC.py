#!/usr/A/anaconda3/envs/py_3.8 python
# -*- coding: UTF-8 -*-
'''
@Project ：BiRC-DeepSC-EUR-8 
@File    ：utils_for_BiMSC.py
@Author  ：Mingtong Zhang
@Date    ：2024/4/18 14:38 
'''
import torch
import numpy as np

from utils.utils_general import Channels, SNR_to_noise


def create_key_masks(src, tgt, padding_idx, device):
    src_key_mask = (src == padding_idx).float().unsqueeze(1)  # [batch, 1, seq_len]
    tgt_key_mask = (tgt == padding_idx).float().unsqueeze(1)  # [batch, 1, seq_len]
    look_ahead_mask = torch.triu(torch.full((tgt.size(1), tgt.size(1)), float('1')), diagonal=1)

    return src_key_mask.to(device), tgt_key_mask.to(device), look_ahead_mask.to(device)


def PowerNormalize(x, key_mask, mask=None, power_tx=1):
    x_square = torch.mul(x, x)

    if mask is not None:
        real_symbol_num = torch.sum(mask, dim=1, keepdim=True)
        power = (torch.sum(x_square, dim=-1, keepdim=True) / real_symbol_num).sqrt()
    else:
        real_symbol_num = torch.sum(key_mask, dim=1, keepdim=True) * x_square.size(-1)
        power = (torch.sum(x_square, dim=(-2, -1), keepdim=True) / real_symbol_num).sqrt()

    x = np.sqrt(power_tx) * torch.div(x, power)

    return x


def get_index_len_sent(value, ranges=None):
    for i, (start, end) in enumerate(ranges):
        if start <= value <= end:
            return i
    return None  # 如果不在任何范围内


def train_step(model, sent, pad, end, snr, opt, loss_function, channel, rate_min, rate_max, device, threshold, fanet_mode):
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

    # 计算出batch中每个句子长度，确定 实际信道特征维度 与 每个样本丢弃的token数目
    batch_dim = src.size(0)
    channel_dim = jsc_enc_output.size(-1)
    len_sent = torch.sum(key_mask, dim=1)
    rate = torch.randint(rate_min, rate_max + 1, (batch_dim, 1)).to(device)

    # 进行特征维度掩码：FANet输出特征维度掩码
    if fanet_mode:
        fd_mask_temp = model.fa_net(jsc_enc_output, rate, noise_std, len_sent, src_key_mask, device, threshold)
        feature_dim = torch.sum(fd_mask_temp, dim=1, keepdim=True).detach()

        indicator = np.random.uniform(0, 1)
        if indicator < 0.01:
            print(f"2*rate = {2 * rate[0].item()}, feature_dim = {feature_dim[0].item()}, SNR = {snr[0].item()} dB")

        for bs in range(batch_dim):
            feature_dim[bs] = torch.max(feature_dim[bs], 2 * rate[bs])
        feature_dim = feature_dim.long()

        fd_mask_temp = fd_mask_temp.unsqueeze(1)
        jsc_enc_output = jsc_enc_output * fd_mask_temp
    else:
        feature_dim = torch.round(2 * (torch.rand_like(len_sent) * (rate_max - rate) + rate)).long()

    fd_mask = torch.ones(batch_dim, 1, channel_dim, device=device)
    for bs in range(batch_dim):
        fd_mask[bs, :, feature_dim[bs, 0]:] = 0

    # 进行词元维度掩码：确定掩码后的词元长度，计算词元维度掩码向量，进行掩码
    token_dim = torch.floor(2 * rate * len_sent / feature_dim).long()
    extra_dim = (torch.round(2 * rate * len_sent) - token_dim * feature_dim).long()
    td_mask = torch.ones(batch_dim, src.size(1), 1, device=device)

    # 确定词元维度掩码向量
    for bs in range(batch_dim):
        td_mask[bs, token_dim[bs, 0]:, :] = 0
    ftd_mask = fd_mask * td_mask
    for bs in range(batch_dim):
        if extra_dim[bs]:
            ftd_mask[bs, token_dim[bs, 0], :extra_dim[bs]] = 1

    # 进行掩码
    jsc_enc_output = jsc_enc_output.reshape(batch_dim, -1)
    ftd_mask = ftd_mask.reshape(batch_dim, -1)
    jsc_enc_output = jsc_enc_output * ftd_mask

    # 功率归一化
    tx_sig = PowerNormalize(jsc_enc_output, key_mask, ftd_mask)

    channels = Channels()
    if channel == 'AWGN':
        rx_sig = channels.AWGN(tx_sig, noise_std, device=device)
    elif channel == 'Rayleigh':
        rx_sig, _ = channels.Rayleigh(tx_sig, noise_std, device=device)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    # 屏蔽padding的影响
    rx_sig = rx_sig * ftd_mask
    # 先Reshape，再联合信源信道译码
    jsc_dec_input = rx_sig.reshape(batch_dim, -1, channel_dim)

    jsc_dec_output = model.jsc_decoder(jsc_dec_input)
    # 语义译码
    dec_output = model.semantic_decoder(tgt_inp, jsc_dec_output, look_ahead_mask=look_ahead_mask, tgt_key_mask=tgt_key_mask, memory_key_mask=src_key_mask)
    pred = model.dense(dec_output)
    # 计算Loss
    ntokens = pred.size(-1)
    loss = loss_function(pred.reshape(-1, ntokens), tgt_real.reshape(-1))

    loss.backward()
    opt.step()

    return loss.item()


def eval_step(model, sent, pad, end, snr, loss_function, channel, rate_min, rate_max, device, threshold, fanet_mode):
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

    # 计算出batch中每个句子长度，确定 实际信道特征维度 与 每个样本丢弃的token数目
    batch_dim = src.size(0)
    channel_dim = jsc_enc_output.size(-1)
    len_sent = torch.sum(key_mask, dim=1)
    rate = torch.randint(rate_min, rate_max + 1, (batch_dim, 1)).to(device)

    # 进行特征维度掩码：FANet输出特征维度掩码
    if fanet_mode:
        fd_mask_temp = model.fa_net(jsc_enc_output, rate, noise_std, len_sent, src_key_mask, device, threshold)
        feature_dim = torch.sum(fd_mask_temp, dim=1, keepdim=True).detach()

        indicator = np.random.uniform(0, 1)
        if indicator < 0.01:
            print(f"2*rate = {2 * rate[0].item()}, feature_dim = {feature_dim[0].item()}, SNR = {snr[0].item()} dB")

        for bs in range(batch_dim):
            feature_dim[bs] = torch.max(feature_dim[bs], 2 * rate[bs])
        feature_dim = feature_dim.long()

        fd_mask_temp = fd_mask_temp.unsqueeze(1)
        jsc_enc_output = jsc_enc_output * fd_mask_temp
    else:
        feature_dim = torch.round(2 * (torch.rand_like(len_sent) * (rate_max - rate) + rate)).long()

    fd_mask = torch.ones(batch_dim, 1, channel_dim, device=device)
    for bs in range(batch_dim):
        fd_mask[bs, :, feature_dim[bs, 0]:] = 0

    # 进行词元维度掩码：确定掩码后的词元长度，计算词元维度掩码向量，进行掩码
    token_dim = torch.floor(2 * rate * len_sent / feature_dim).long()
    extra_dim = (torch.round(2 * rate * len_sent) - token_dim * feature_dim).long()
    td_mask = torch.ones(batch_dim, src.size(1), 1, device=device)

    # 确定词元维度掩码向量
    for bs in range(batch_dim):
        td_mask[bs, token_dim[bs, 0]:, :] = 0
    ftd_mask = fd_mask * td_mask
    for bs in range(batch_dim):
        if extra_dim[bs]:
            ftd_mask[bs, token_dim[bs, 0], :extra_dim[bs]] = 1

    # 进行掩码
    jsc_enc_output = jsc_enc_output.reshape(batch_dim, -1)
    ftd_mask = ftd_mask.reshape(batch_dim, -1)
    jsc_enc_output = jsc_enc_output * ftd_mask

    # 功率归一化
    tx_sig = PowerNormalize(jsc_enc_output, key_mask, ftd_mask)

    channels = Channels()
    if channel == 'AWGN':
        rx_sig = channels.AWGN(tx_sig, noise_std, device=device)
    elif channel == 'Rayleigh':
        rx_sig, _ = channels.Rayleigh(tx_sig, noise_std, device=device)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    # 屏蔽padding的影响
    rx_sig = rx_sig * ftd_mask
    # 先Reshape，再联合信源信道译码
    jsc_dec_input = rx_sig.reshape(batch_dim, -1, channel_dim)

    jsc_dec_output = model.jsc_decoder(jsc_dec_input)
    # 语义译码
    dec_output = model.semantic_decoder(tgt_inp, jsc_dec_output, look_ahead_mask=look_ahead_mask, tgt_key_mask=tgt_key_mask, memory_key_mask=src_key_mask)
    pred = model.dense(dec_output)
    # 计算Loss
    ntokens = pred.size(-1)
    loss = loss_function(pred.reshape(-1, ntokens), tgt_real.reshape(-1))

    return loss.item()


def greedy_decode(model, sent, pad, end, snr, len_max, start_symbol, channel, rate, device, threshold):
    model.eval()
    noise_std = SNR_to_noise(snr)

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

    # 计算出batch中每个句子长度，和每个样本的速率
    batch_dim = src.size(0)
    channel_dim = jsc_enc_output.size(-1)
    len_sent = torch.sum(key_mask, dim=1)
    rate = torch.zeros_like(len_sent) + torch.tensor([rate]).to(device)

    # 进行特征维度掩码：确定掩码后的特征长度，计算特征维度掩码向量，进行掩码
    fd_mask = model.fa_net(jsc_enc_output, rate, noise_std, len_sent, src_key_mask, device, threshold)
    feature_dim = torch.sum(fd_mask, dim=1, keepdim=True)

    indicator = np.random.uniform(0, 1)
    bs_random = np.random.randint(0, batch_dim)
    if indicator < 0.05:
        print(f"2*rate = {2 * rate[0].item()}, len_sent = {len_sent[bs_random].item()}, feature_dim = {feature_dim[bs_random].item()}, SNR = {snr[0].item()} dB")

    for bs in range(batch_dim):
        feature_dim[bs] = torch.max(feature_dim[bs], 2 * rate[bs])
    feature_dim = feature_dim.long()

    fd_mask = torch.ones(batch_dim, 1, channel_dim, device=device)
    for bs in range(batch_dim):
        fd_mask[bs, :, feature_dim[bs, 0]:] = 0
    jsc_enc_output = jsc_enc_output * fd_mask

    # 进行词元维度掩码：确定掩码后的词元长度，计算词元维度掩码向量，进行掩码
    token_dim = torch.floor(2 * rate * len_sent / feature_dim).long()
    extra_dim = (torch.round(2 * rate * len_sent) - token_dim * feature_dim).long()
    td_mask = torch.ones(batch_dim, src.size(1), 1, device=device)

    # 确定词元维度掩码向量
    for bs in range(batch_dim):
        td_mask[bs, token_dim[bs, 0]:, :] = 0
    ftd_mask = fd_mask * td_mask
    for bs in range(batch_dim):
        if extra_dim[bs]:
            ftd_mask[bs, token_dim[bs, 0], :extra_dim[bs]] = 1

    # 进行掩码
    jsc_enc_output = jsc_enc_output.reshape(batch_dim, -1)
    ftd_mask = ftd_mask.reshape(batch_dim, -1)
    jsc_enc_output = jsc_enc_output * ftd_mask

    # 功率归一化
    tx_sig = PowerNormalize(jsc_enc_output, key_mask, ftd_mask)

    channels = Channels()
    if channel == 'AWGN':
        rx_sig = channels.AWGN(tx_sig, noise_std, device=device)
    elif channel == 'Rayleigh':
        rx_sig, _ = channels.Rayleigh(tx_sig, noise_std, device=device)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    # 屏蔽padding的影响
    rx_sig = rx_sig * ftd_mask
    # 先Reshape，再联合信源信道译码
    jsc_dec_input = rx_sig.reshape(batch_dim, -1, channel_dim)
    memory = model.jsc_decoder(jsc_dec_input)

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
