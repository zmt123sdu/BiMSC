#!/usr/A/anaconda3/envs/py_3.8 python
# -*- coding: UTF-8 -*-
'''
@Project ：BiRC-DeepSC-EUR-8 
@File    ：utils_general.py
@Author  ：Mingtong Zhang
@Date    ：2024/9/24 10:49 
'''
import collections
import os
import math
import time
import random
import numpy as np
import torch
import torch.nn as nn

from matplotlib import pyplot as plt
from IPython import display
from scipy import io
from w3lib.html import remove_tags
from nltk.translate.bleu_score import sentence_bleu


def initNetParams(model):
    '''Init net parameters.'''
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal_(p)
    return model


def SNR_to_noise(snr, power_tx=1):
    snr = 10 ** (snr / 10)
    noise_std = torch.sqrt(power_tx / snr)

    return noise_std


def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def count_corpus(tokens):
    """Count token frequencies."""
    # Here `tokens` is a 1D list or 2D list
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Flatten a list of token lists into a list of tokens
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib."""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def load_config(filename='config.txt'):
    """
    从配置文件加载参数，支持数值列表。

    参数：
    filename (str): 配置文件的名称，默认是 'config.txt'。

    返回：
    dict: 加载的配置参数。
    """
    config = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            # 跳过注释行
            if line.startswith("#") or not line:
                continue
            key, value = line.split(' = ')
            # 检查是否是数值列表
            if ',' in value:
                value = list(map(float, value.split(', ')))  # 将字符串转换为浮点数列表
                if all(val.is_integer() for val in value):  # 如果全是整数，转换为整数列表
                    value = list(map(int, value))
            elif value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                value = int(value)
            elif value.replace('.', '', 1).isdigit() or (value.startswith('-') and value[1:].replace('.', '', 1).isdigit()):
                value = float(value)
            elif value == 'True':
                value = True
            elif value == 'False':
                value = False
            elif value == 'None':
                value = None  # 将字符串 'None' 转换为 None
            config[key] = value
    return config


def bleu(pred_seq, label_seq, k=4, weights=(1, 0, 0, 0)):  # @save
    """计算BLEU"""
    pred_tokens, label_tokens = pred_seq, label_seq
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / max(1, len_pred)))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[''.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[''.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[''.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / max(1, (len_pred - n + 1)), weights[n - 1])
    return score


class Vocab:
    """Vocabulary for text."""

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # Sort according to frequencies
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 将预定义的特殊字符加入字典
        self.token_to_idx = reserved_tokens
        self.idx_to_token = [token for idx, token in enumerate(self.token_to_idx)]
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # Index for the unknown token
        return 3

    @property
    def token_freqs(self):  # Index for the unknown token
        return self._token_freqs


class Timer:
    """Record multiple running times."""

    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()

    def time_show(self, time_sec, mode=1):
        if mode == 1:
            if time_sec > 3600:
                hour = int(time_sec / 3600)
                min = int((time_sec - 3600 * hour) / 60)
                sec = time_sec - 3600 * hour - 60 * min
                time_description = f'{hour} hour {min} min {sec:.2f} sec'
            elif time_sec > 60:
                min = int(time_sec / 60)
                sec = time_sec - 60 * min
                time_description = f'{min} min {sec:.2f} sec'
            else:
                sec = time_sec
                time_description = f'{sec:.2f} sec'
        else:
            time_description = f'{time_sec,:.2f} sec'
        return time_description


class Animator:
    """For plotting data in animation."""

    def __init__(self, x_label=None, y_label=None, legend=None, x_lim=None, y_lim=None, x_scale='linear',
                 y_scale='linear', fmts=('-', 'm--', 'g-.', 'r:'), n_rows=1, n_cols=1, fig_size=(3.5 * 6, 2.5 * 6)):
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
        if n_rows * n_cols == 1:
            self.axes = [self.axes, ]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: set_axes(self.axes[0], x_label, y_label, x_lim, y_lim, x_scale, y_scale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        plt.draw()
        plt.pause(0.01)
        display.clear_output(wait=True)

    def pause(self):
        plt.show()

    def save(self, filename):
        self.config_axes()
        plt.grid()
        plt.savefig(filename)


class Data_savemat:
    """将数据保存为mat格式"""

    def __init__(self):
        self.x_data = []
        self.y_data = []
        self.legend = []

    def append(self, x=1, y=1, legend=None):
        self.x_data.append(x)
        self.y_data.append(y)
        if legend:
            self.legend.append(legend)
            self.x_data.pop()
            self.y_data.pop()

    def savemat(self, filename):
        io.savemat(filename, {'x_data': self.x_data, 'y_data': self.y_data, 'legend': self.legend})


class Channels():

    def AWGN(self, Tx_sig, n_std, device):
        if not torch.is_tensor(n_std):
            n_std = torch.tensor(n_std, device=device)
        noise = torch.randn_like(Tx_sig) * n_std.reshape(-1, *([1] * (Tx_sig.dim() - 1)))
        noise = noise.to(dtype=Tx_sig.dtype, device=device)
        Rx_sig = Tx_sig + noise
        return Rx_sig

    def Rayleigh(self, Tx_sig, n_std, device):
        shape = Tx_sig.shape
        H_real = torch.normal(0, math.sqrt(1 / 2), size=[shape[0]])
        H_imag = torch.normal(0, math.sqrt(1 / 2), size=[shape[0]])
        H = torch.zeros(size=[shape[0], 2, 2])
        H[:, 0, 0] = H_real
        H[:, 0, 1] = H_imag
        H[:, 1, 0] = -H_imag
        H[:, 1, 1] = H_real
        H_abs = (H_real ** 2 + H_imag ** 2).sqrt().to(device)
        H = H.to(device)
        flag = Tx_sig.size(-1) % 2
        if flag:
            zeros_padding = torch.zeros(shape[0], 1, device=device)
            Tx_sig = torch.concat((Tx_sig, zeros_padding), dim=-1)
        Tx_sig = Tx_sig.view(shape[0], -1, 2)
        Tx_sig = torch.matmul(Tx_sig, H)
        Rx_sig = self.AWGN(Tx_sig, n_std, device=device)
        # Channel estimation
        if len(shape) == 2:
            Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape[0], -1)
        else:
            Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)
        if flag:
            Rx_sig = Rx_sig[:, :-1]

        return Rx_sig, H_abs


class BleuScore():
    def __init__(self, w1, w2, w3, w4):
        self.w1 = w1  # 1-gram weights
        self.w2 = w2  # 2-grams weights
        self.w3 = w3  # 3-grams weights
        self.w4 = w4  # 4-grams weights

    def compute_blue_score(self, predicted, real, handcraft_mode=True):
        score = []
        for (sent1, sent2) in zip(predicted, real):
            sent1 = remove_tags(sent1).split()
            sent2 = remove_tags(sent2).split()
            if handcraft_mode:
                score.append(bleu(sent1, sent2, weights=(self.w1, self.w2, self.w3, self.w4)))
            else:
                score.append(sentence_bleu([sent1], sent2, weights=(self.w1, self.w2, self.w3, self.w4)))
        return score


class SeqtoText:
    def __init__(self, vocb_dictionary, end_idx):
        self.reverse_word_map = dict(zip(vocb_dictionary.values(), vocb_dictionary.keys()))
        self.end_idx = end_idx

    def sequence_to_text(self, list_of_indices):
        # Looking up words in dictionary
        words = []
        for idx in list_of_indices:
            if idx == self.end_idx:
                break
            else:
                words.append(self.reverse_word_map.get(idx))
        words = ' '.join(words)
        return (words)


def get_required_epoch(path, order=-1):
    idx_list = []
    for fn in os.listdir(path):
        if not fn.endswith('.pth'): continue
        idx = int(os.path.splitext(fn)[0].split('_')[-1])  # read the idx of image
        idx_list.append((os.path.join(path, fn), idx))

    idx_list.sort(key=lambda x: x[1])  # sort the image by the idx
    _, epoch = idx_list[order]

    return epoch
