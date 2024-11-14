#!/usr/A/anaconda3/envs/py_3.8 python
# -*- coding: UTF-8 -*-
'''
@Project ：BiRC-DeepSC-EUR-8 
@File    ：config_train.py
@Author  ：Mingtong Zhang
@Date    ：2024/4/15 22:37 
'''
import argparse
from utils.utils_general import load_config


def get_train_parser(case, channel, config_number, d_channel=32):
    config_file_path = f'./config/train_config_{config_number}.txt'
    config_dict = load_config(config_file_path)

    parser = get_mian_parser(channel=channel, snr_min=config_dict['snr_min'], snr_max=config_dict['snr_max'], rate_min=config_dict['rate_min'], rate_max=config_dict['rate_max'],
                             fixed_padding=config_dict['fixed_padding'], seed=config_dict['seed'], lr_initial=config_dict['lr_initial'],
                             d_channel_max=config_dict['d_channel_max'], epochs=config_dict['epochs'], batch_size=config_dict['batch_size'], optimizer=config_dict['optimizer'],
                             len_max=config_dict['len_max'], d_model=config_dict['d_model'], dff=config_dict['dff'], num_layers=config_dict['num_layers'],
                             num_heads=config_dict['num_heads'], dropout=config_dict['dropout'], scheduler_type=config_dict['lr_scheduler_type'],
                             decay_step_list=config_dict['decay_step_list'], lr_decay_rate=config_dict['lr_decay_rate'])

    if case == 'DeepSC':
        train_info = f'{case}_{channel}_train-rate({d_channel / 2})_train-config({config_number})'
        parser = get_parser_set1(parser, train_info=train_info)
    elif case == 'FD-DeepSC':
        train_info = f'{case}_{channel}_train-config({config_number})'
        parser = get_parser_set1(parser, train_info=train_info)
    elif case == 'TD-DeepSC':
        train_info = f'{case}_{channel}_train-config({config_number})'
        parser = get_parser_set1(parser, train_info=train_info)
    elif case == 'BiMSC':
        train_info = f'{case}_{channel}_train-config({config_number})'
        parser = get_parser_set2(parser, train_info=train_info, mlp_ratio=config_dict['mlp_ratio'], stage1_epoch=config_dict['stage1_epoch'], threshold=config_dict['threshold'])

    return parser


def get_parser_set1(parser, train_info):
    parser.add_argument('--checkpoint-path', default=f'./checkpoints/{train_info}', type=str)
    parser.add_argument('--best-save-path', default=f'./checkpoints/{train_info}_best', type=str)
    parser.add_argument('--train-info', default=train_info, type=str)

    return parser


def get_parser_set2(parser, train_info, mlp_ratio, stage1_epoch, threshold):
    parser.add_argument('--checkpoint-path', default=f'./checkpoints/{train_info}', type=str)
    parser.add_argument('--best-save-path', default=f'./checkpoints/{train_info}_best', type=str)
    parser.add_argument('--train-info', default=train_info, type=str)
    parser.add_argument('--mlp_ratio', default=mlp_ratio, type=int)
    parser.add_argument('--stage1_epoch', default=stage1_epoch, type=int)
    parser.add_argument('--threshold', default=threshold, type=float)

    return parser


def get_mian_parser(channel, d_channel_max, snr_min, snr_max, rate_min, rate_max, fixed_padding, lr_initial, epochs, seed, batch_size, len_max, d_model, dff, num_layers,
                    optimizer, num_heads, dropout, scheduler_type, decay_step_list, lr_decay_rate):
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
    parser.add_argument('--snr-min', default=snr_min, type=float)
    parser.add_argument('--snr-max', default=snr_max, type=float)
    # Dataset parameters
    parser.add_argument('--len-max', default=len_max, type=int)
    parser.add_argument('--fixed-padding', default=fixed_padding, type=bool)
    # Training set parameters
    parser.add_argument('--seed', default=seed, type=int)
    parser.add_argument('--batch-size', default=batch_size, type=int)
    parser.add_argument('--lr', default=lr_initial, type=float)
    parser.add_argument('--scheduler_type', default=scheduler_type, type=str)
    parser.add_argument('--decay_step_list', default=decay_step_list, nargs='+')
    parser.add_argument('--lr_decay_rate', default=lr_decay_rate, type=float)
    parser.add_argument('--optimizer', default=optimizer, type=str)
    parser.add_argument('--epochs', default=epochs, type=int)
    parser.add_argument('--rate-min', default=rate_min, type=int)
    parser.add_argument('--rate-max', default=rate_max, type=int)

    return parser
