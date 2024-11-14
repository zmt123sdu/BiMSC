#!/usr/A/anaconda3/envs/py_3.8 python
# -*- coding: UTF-8 -*-
'''
@Project ：BiRC-DeepSC-EUR-8 
@File    ：dataset.py
@Author  ：Mingtong Zhang
@Date    ：2024/2/13 17:12 
'''
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


class EurDataset(Dataset):
    def __init__(self, data_path='train'):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

    def __getitem__(self, index):
        sents = self.data[index]
        return sents

    def __len__(self):
        return len(self.data)


class collater():
    def __init__(self, fixed_length_padding=False, len_max=32, pair_mode=False):
        self.fixed_length_padding = fixed_length_padding
        self.len_max = len_max
        self.pair_mode = pair_mode

    def __call__(self, batch):

        batch_size = len(batch)

        if self.fixed_length_padding:
            """使每个句子都padding到最大固定长度"""
            len_max = self.len_max
            sents = np.zeros((batch_size, len_max), dtype=np.int64)
            sort_by_len = sorted(batch, key=lambda x: len(x), reverse=True)

            for i, sent in enumerate(sort_by_len):
                length = len(sent)
                sents[i, :length] = sent  # padding the questions
        elif self.pair_mode:
            """使每个句子都padding到最大固定长度"""
            len_max = self.len_max
            paragraph_length = len(batch[0])
            sents = np.zeros((batch_size, len_max, paragraph_length), dtype=np.int64)
            sort_by_len = sorted(batch, key=lambda x: len(x[0]), reverse=True)

            for j in range(paragraph_length):
                for i, sent in enumerate(sort_by_len):
                    length = len(sent[j])
                    sents[i, :length, j] = sent[j]  # padding the questions
        else:
            """使每个batch中的每个句子都padding到当前batch最大长度"""
            len_max = max(map(lambda x: len(x), batch))  # get the max length of tgt sentence in current batch
            sents = np.zeros((batch_size, len_max), dtype=np.int64)
            sort_by_len = sorted(batch, key=lambda x: len(x), reverse=True)

            for i, sent in enumerate(sort_by_len):
                length = len(sent)
                sents[i, :length] = sent  # padding the questions

        return torch.from_numpy(sents)
