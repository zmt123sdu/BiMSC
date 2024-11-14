#!/usr/A/anaconda3/envs/py_3.8 python
# -*- coding: UTF-8 -*-
'''
@Project ：BiRC-DeepSC-EUR-8 
@File    ：config_test.py
@Author  ：Mingtong Zhang
@Date    ：2024/4/15 22:38 
'''
import argparse
from utils.utils_general import get_required_epoch, load_config


def get_test_parser(case, channel, train_config_number, test_config_number, test_rate_list, trained_epoch=200, best_mode=False, d_channel=32):
    config_file_path = f'./config/test_config_{test_config_number}.txt'
    config_dict = load_config(config_file_path)

    parser = get_mian_parser(channel=channel, test_loop=config_dict['test_loop'], test_bs=config_dict['test_batch_size'], seed=config_dict['seed'],
                             fixed_padding=config_dict['fixed_padding'], dropout=config_dict['dropout'], d_channel_max=config_dict['d_channel_max'],
                             d_model=config_dict['d_model'], dff=config_dict['dff'], num_heads=config_dict['num_heads'], num_layers=config_dict['num_layers'],
                             len_max=config_dict['len_max'])

    if case == 'DeepSC':
        train_info = f'{case}_{channel}_train-rate({d_channel / 2})_train-config({test_config_number})'
        trained_fold = f'./checkpoints/{train_info}'
        if best_mode:
            trained_fold = f'./checkpoints/{train_info}_best'
            trained_epoch = get_required_epoch(trained_fold)
        trained_path = f'{trained_fold}/epoch_{trained_epoch}.pth'
        test_info_list = []
        for test_rate in test_rate_list:
            test_info = train_info + f'_test-config({test_config_number})_test-rate({test_rate})_trained-epoch({trained_epoch})'
            test_info_list.append(test_info)
        parser = get_parser_set1(parser, test_info_list=test_info_list, test_rate_list=test_rate_list, trained_path=trained_path)

    elif case == 'FD-DeepSC':
        train_info = f'{case}_{channel}_train-config({train_config_number})'
        trained_fold = f'./checkpoints/{train_info}'
        if best_mode:
            trained_fold = f'./checkpoints/{train_info}_best'
            trained_epoch = get_required_epoch(trained_fold)
        trained_path = f'{trained_fold}/epoch_{trained_epoch}.pth'
        test_info_list = []
        for test_rate in test_rate_list:
            test_info = train_info + f'_test-config({test_config_number})_test-rate({test_rate})_trained-epoch({trained_epoch})'
            test_info_list.append(test_info)
        parser = get_parser_set1(parser, test_info_list=test_info_list, test_rate_list=test_rate_list, trained_path=trained_path)

    elif case == 'TD-DeepSC':
        train_info = f'{case}_{channel}_train-config({train_config_number})'
        trained_fold = f'./checkpoints/{train_info}'
        if best_mode:
            trained_fold = f'./checkpoints/{train_info}_best'
            trained_epoch = get_required_epoch(trained_fold)
        trained_path = f'{trained_fold}/epoch_{trained_epoch}.pth'
        test_info_list = []
        for test_rate in test_rate_list:
            test_info = train_info + f'_test-config({test_config_number})_test-rate({test_rate})_trained-epoch({trained_epoch})'
            test_info_list.append(test_info)
        parser = get_parser_set1(parser, test_info_list=test_info_list, test_rate_list=test_rate_list, trained_path=trained_path)

    elif case == 'BiMSC':
        train_info = f'{case}_{channel}_train-config({train_config_number})'
        trained_fold = f'./checkpoints/{train_info}'
        if best_mode:
            trained_fold = f'./checkpoints/{train_info}_best'
            trained_epoch = get_required_epoch(trained_fold)
        trained_path = f'{trained_fold}/epoch_{trained_epoch}.pth'
        test_info_list = []
        for test_rate in test_rate_list:
            test_info = train_info + f'_test-config({test_config_number})_test-rate({test_rate})_trained-epoch({trained_epoch})'
            test_info_list.append(test_info)
        parser = get_parser_set2(parser, test_info_list=test_info_list, test_rate_list=test_rate_list, trained_path=trained_path, threshold=config_dict['threshold'],
                                 mlp_ratio=config_dict['mlp_ratio'])

    return parser


def get_parser_set1(parser, test_info_list, test_rate_list, trained_path):
    parser.add_argument('--trained-path', default=trained_path, type=str)
    parser.add_argument('--test-info-list', default=test_info_list, nargs='+')
    parser.add_argument('--test-rate-list', default=test_rate_list, nargs='+')

    return parser


def get_parser_set2(parser, test_info_list, test_rate_list, trained_path, threshold, mlp_ratio):
    parser.add_argument('--trained-path', default=trained_path, type=str)
    parser.add_argument('--test-info-list', default=test_info_list, nargs='+')
    parser.add_argument('--test-rate-list', default=test_rate_list, nargs='+')
    parser.add_argument('--mlp_ratio', default=mlp_ratio, type=int)
    parser.add_argument('--threshold', default=threshold, type=float)

    return parser


def get_mian_parser(channel, d_channel_max, fixed_padding, d_model, dff, num_layers, num_heads, dropout, len_max, seed, test_loop, test_bs):
    parser = argparse.ArgumentParser()
    # File path parameters
    parser.add_argument('--train-data-path', default='./data/europarl/train_data.pkl', type=str)
    parser.add_argument('--test-data-path', default='./data/europarl/test_data.pkl', type=str)
    parser.add_argument('--vocab-file', default='./data/europarl/vocab.pkl', type=str)
    # Model basic parameters
    parser.add_argument('--d-model', default=d_model, type=int)
    parser.add_argument('--d_channel_max', default=d_channel_max, type=int)
    parser.add_argument('--dff', default=dff, type=int)
    parser.add_argument('--num-layers', default=num_layers, type=int)
    parser.add_argument('--num-heads', default=num_heads, type=int)
    parser.add_argument('--dropout', default=dropout, type=float)
    # Channel simulation parameters
    parser.add_argument('--channel', default=channel, type=str)
    # Dataset parameters
    parser.add_argument('--len-max', default=len_max, type=int)
    parser.add_argument('--fixed-padding', default=fixed_padding, type=bool)
    # Training set parameters
    parser.add_argument('--seed', default=seed, type=int)
    parser.add_argument('--test-loop', default=test_loop, type=int)
    parser.add_argument('--test-bs', default=test_bs, type=int)

    return parser
