#!/usr/A/anaconda3/envs/py_3.8 python
# -*- coding: UTF-8 -*-
'''
@Project ：BiRC-DeepSC-EUR-8 
@File    ：text_preprocess.py
@Author  ：Mingtong Zhang
@Date    ：2024/2/13 10:24 
'''
import argparse
from tqdm import tqdm
from matplotlib import pyplot as plt
from utils.utils_general import Vocab
import os
import unicodedata
import pickle
import re
from w3lib.html import remove_tags

len_sent_min = 26
len_sent_max = 30
parser = argparse.ArgumentParser()
parser.add_argument('--input-data-dir', default='europarl/en', type=str)
parser.add_argument('--output-train-dir', default=f'europarl/train_data_[{len_sent_min}, {len_sent_max}].pkl', type=str)
parser.add_argument('--output-test-dir', default=f'europarl/test_data_[{len_sent_min}, {len_sent_max}].pkl', type=str)
parser.add_argument('--output-vocab', default='europarl/vocab.pkl', type=str)
parser.add_argument('--len_sent_min', default=len_sent_min, type=int)
parser.add_argument('--len_sent_max', default=len_sent_max, type=int)


def unicode_to_ascii(s):
    """使用unicodedata模块将字符串标准化，NFD指定字符串标准化的方式为分解为多个组合字符表示
    unicodedata.category把一个字符返回它在UNICODE里分类的类型。 Mn表示 Mark, Nonspacing"""
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def normalize_string(s):
    # normalize unicode characters
    s = unicode_to_ascii(s)
    # remove the XML-tags
    s = remove_tags(s)
    # add white space before ;,!.?   \1表示匹配到的字符
    s = re.sub(r'([;,!.?])', r' \1', s)
    # 将所有非字母和非.!?的字符全部替换为空格，并删除两边空白
    s = re.sub(r'[^a-zA-Z;,.!?]+', r' ', s).strip()
    # \S 匹配所有空白字符，并替换为空格
    s = re.sub(r'\s+', r' ', s)
    # change to lower letter
    s = s.lower()
    return s


def cutted_data(cleaned, MIN_LENGTH=4, MAX_LENGTH=30):
    """对输入的每行句子进行判断，只保留长度大于等于MIN_LENGTH，小于等于MAX_LENGTH的句子"""
    cutted_lines = list()
    for line in cleaned:
        length = len(line.split())
        if length >= MIN_LENGTH and length <= MAX_LENGTH:
            line = [word for word in line.split()]
            cutted_lines.append(' '.join(line))
    return cutted_lines


def process(text_path):
    """处理文本文件，"""
    fop = open(text_path, 'r', encoding='utf8')
    raw_data = fop.read()
    # 删除两边空白，并以换行符将文本文件内容分割，也就是每一行为列表中的一个元素
    sentences = raw_data.strip().split('\n')
    sentences_new = []
    # 直接以换行符分割文本得到的长度合适的样本太少，进一步以'.'和'?'对样本进行分割
    for raw_sentence in sentences:
        sentence_dot = raw_sentence.split('.')
        if len(sentence_dot) >= 2:
            for i in range(len(sentence_dot) - 1):
                sentence_dot[i] = sentence_dot[i] + '.'
        # 以‘？’进行分割
        for raw_sentence_ques in sentence_dot:
            sentence_ques = raw_sentence_ques.split('?')
            if len(sentence_ques) >= 2:
                for i in range(len(sentence_ques) - 1):
                    sentence_ques[i] = sentence_ques[i] + '?'
            sentences_new = sentences_new + sentence_ques

    # 对文本数据进行clean
    raw_data_input = [normalize_string(data) for data in sentences_new]
    # 挑选出符合长度的句子
    raw_data_input = cutted_data(raw_data_input, MIN_LENGTH=args.len_sent_min, MAX_LENGTH=args.len_sent_max)
    fop.close()
    return raw_data_input


def build_vocab(sequences, min_token_count=3):
    # sequences：句子列表，每一个列表元素是一个句子
    # min_token_count：出现频次大于等于min_token_count的token才会被加入到词汇表token_to_idx
    # 返回vocab类：包含token_to_idx和idx_to_token字典、词频统计及to_tokens函数

    tokens = [token for seq in sequences for token in seq.split()]
    vocab = Vocab(tokens=tokens, min_freq=min_token_count, reserved_tokens={'<PAD>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3, })

    # 画出词频图
    freqs = [freq for token, freq in vocab.token_freqs]
    fig2, axes = plt.subplots()
    axes.plot(freqs)
    axes.set_xlabel('token: x')
    axes.set_ylabel('frequency: n(x)')
    axes.set_xscale('log')
    axes.set_yscale('log')
    plt.show()

    return vocab


def tokenize(s, delim=' ', add_start_token=True, add_end_token=True, punct_to_keep=None, punct_to_remove=None):
    """
    Tokenize a sequence, converting a string s into a list of (string) tokens by
    splitting on the specified delimiter. Optionally keep or remove certain
    punctuation marks and add start and end tokens.
    s：输入的句子字符串
    delim：分隔符，默认为空格
    add_start_token：是否添加start_token，默认添加
    add_end_token：是否添加end_token，默认添加
    punct_to_keep:要保留的标点符号，默认无
    punct_to_remove：要移除的标点符号，默认无
    最后返回tokens：一个tokens的列表
    """
    if punct_to_keep is not None:
        for p in punct_to_keep:
            # 在要保留的标点符号前加一个指定的分隔符
            s = s.replace(p, '%s%s' % (delim, p))

    if punct_to_remove is not None:
        for p in punct_to_remove:
            # 将要移出的标点符号移除
            s = s.replace(p, '')

    tokens = s.split()
    if add_start_token:
        tokens.insert(0, '<START>')
    if add_end_token:
        tokens.append('<END>')
    return tokens


def main(args):
    data_dir = './data/'
    args.input_data_dir = data_dir + args.input_data_dir
    args.output_train_dir = data_dir + args.output_train_dir
    args.output_test_dir = data_dir + args.output_test_dir
    args.output_vocab = data_dir + args.output_vocab

    print('Preprocess Raw Text')

    sentences = []
    for fn in tqdm(sorted(os.listdir(args.input_data_dir))):
        if not fn.endswith('.txt'): continue
        process_sentences = process(os.path.join(args.input_data_dir, fn))  # 返回一个list，每个元素是一个经过预处理的句子
        sentences += process_sentences

    # remove the same sentences
    a = {}
    for set in sentences:
        if set not in a:
            a[set] = 0
        a[set] += 1
    sentences = list(a.keys())  # 由句子构成的列表，每个句子是列表的一个元素
    print('Number of sentences: {}'.format(len(sentences)))

    # 绘制一个直方图，显示每个⽂本序列所包含的词元数量
    plt.figure(1)
    plt.ylabel('length of sentence')
    plt.xlabel('frequency number')
    _, _, patches = plt.hist([len(l.split()) for l in sentences], bins=16, rwidth=0.8, range=(4, 30), align='left', label='source')
    plt.legend(loc='upper right')
    plt.show()

    print('Build Vocab')
    vocab = build_vocab(sentences, min_token_count=3)
    print('Number of words in Vocab: {}'.format(len(vocab)))

    # save the vocab
    if args.output_vocab != '':
        with open(args.output_vocab, 'wb') as f:
            pickle.dump(vocab, f)

    print('Start encoding txt')
    results = []
    for seq in tqdm(sentences):
        words = tokenize(seq)  # 给每句话加上了start和end
        tokens = [vocab[word] for word in words]
        results.append(tokens)

    print('Writing Data')
    train_data = results[: round(len(results) * 0.9)]
    test_data = results[round(len(results) * 0.9):]
    train_small_data = train_data[: round(len(train_data) * 0.02)]
    test_small_data = test_data[: round(len(test_data) * 0.02)]

    with open(args.output_train_dir, 'wb') as f:
        pickle.dump(train_data, f)
    with open(args.output_test_dir, 'wb') as f:
        pickle.dump(test_data, f)
    # with open('./data/europarl/train_small_data.pkl', 'wb') as f:
    #     pickle.dump(train_small_data, f)
    # with open('./data/europarl/test_small_data.pkl', 'wb') as f:
    #     pickle.dump(test_small_data, f)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
