#!/usr/A/anaconda3/envs/py_3.8 python
# -*- coding: UTF-8 -*-
'''
@Project ：BiRC-DeepSC-EUR-8 
@File    ：TD-DeepSC-train.py
@Author  ：Mingtong Zhang
@Date    ：2024/2/24 20:11 
'''
import os
import pickle
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import EurDataset, collater
from models.DeepSC import DeepSC
from utils.utils_general import setup_seed, Timer, initNetParams, Animator, Data_savemat
from utils.utils_for_TD_DeepSC import train_step, eval_step
from config.train_config import get_train_parser

# Some global simulation parameters for debugging
channel = 'Rayleigh'
config_number = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = get_train_parser(case='TD-DeepSC', channel=channel, config_number=config_number)
torch.set_printoptions(precision=4)


def train_eval(model, train_dataloader, eval_dataloader, device, num_epoch, lr, optim, init, scheduler_type, save_interval, eval_interval, save_path, resume):
    # 初始化网络参数
    if init:
        initNetParams(model)

    # 将模型发送到对应的设备
    print('training on:', device)
    model.to(device)

    # 优化器选择
    params = (param for param in model.parameters() if param.requires_grad)
    if optim == 'sgd':
        optimizer = torch.optim.SGD(params, lr=lr)
    elif optim == 'adam':
        optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.98), weight_decay=5e-4)
    else:
        optimizer = torch.optim.AdamW(params, lr=lr)

    # 损失函数确定
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction='mean')

    # 学习率管理
    if scheduler_type == 'MultiStep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.decay_step_list, gamma=args.lr_decay_rate)
    elif scheduler_type == 'Exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay_rate)
    elif scheduler_type == 'Step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=args.lr_decay_rate)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=1.0)

    start_epoch = 1
    if resume is not None:
        #  加载之前训过的模型的参数文件
        print(f"loading from {resume}")
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    animator = Animator(x_label='epoch', x_lim=[1, args.epochs], legend=['train loss', 'eval loss'], fig_size=(7, 5))
    data = Data_savemat()
    timer.start()
    best_eval_loss = 100

    for epoch_index in range(start_epoch, num_epoch + 1):
        print("-----------第 {} 轮训练开始------------".format(epoch_index))
        train_loss = []
        timer_epoch = Timer()
        timer_epoch.start()
        train_bar = tqdm(train_dataloader, ncols=100)

        for sents in train_bar:
            snr = torch.randint(args.snr_min, args.snr_max + 1, (sents.size(0), 1))

            snr = snr.to(device)
            sents = sents.to(device)

            batch_loss_train = train_step(model=model, sent=sents, pad=pad_idx, end=end_idx, snr=snr, opt=optimizer, loss_function=loss_fn,
                                          channel=args.channel, rate_min=args.rate_min, rate_max=args.rate_max, device=device)
            train_bar.set_description(f"Epoch: {epoch_index}; Type: Train; Loss: {batch_loss_train:.5f}")

            train_loss.append(batch_loss_train)

        if epoch_index % save_interval == 0:
            os.makedirs(save_path, exist_ok=True)
            save_file = os.path.join(save_path, f"epoch_{epoch_index}.pth")
            torch.save({
                'epoch': epoch_index,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_file)
            print(f"\ncheckpoint has been saved in {save_file}")

        if epoch_index % eval_interval == 0:
            eval_loss = []
            eval_bar = tqdm(eval_dataloader, ncols=100)
            with torch.no_grad():
                for sents_eval in eval_bar:
                    snr = torch.randint(args.snr_min, args.snr_max + 1, (sents_eval.size(0), 1))

                    snr = snr.to(device)
                    sents_eval = sents_eval.to(device)

                    batch_loss_eval = eval_step(model=model, sent=sents_eval, pad=pad_idx, end=end_idx, snr=snr,
                                                loss_function=loss_fn, channel=args.channel, rate_min=args.rate_min, rate_max=args.rate_max, device=device)
                    eval_loss.append(batch_loss_eval)

                    eval_bar.set_description(f"Epoch: {epoch_index}; Type: Eval; Loss: {batch_loss_eval:.5f}")

            animator.add(epoch_index, (sum(train_loss) / len(train_loss), sum(eval_loss) / len(eval_loss)))
            data.append(x=epoch_index, y=(sum(train_loss) / len(train_loss), sum(eval_loss) / len(eval_loss)))
            print(f'train loss: {sum(train_loss) / len(train_loss)}; eval loss: {sum(eval_loss) / len(eval_loss)}')

            average_eval_loss = sum(eval_loss) / len(eval_loss)
            if average_eval_loss < best_eval_loss:
                os.makedirs(args.best_save_path, exist_ok=True)
                save_file = os.path.join(args.best_save_path, f"epoch_{epoch_index}.pth")
                torch.save({'model_state_dict': model.state_dict(), }, save_file)
                print(f'The best Model is saved at epoch-{epoch_index}')
                best_eval_loss = average_eval_loss

        scheduler.step()
        print(f"the training has taken {timer.time_show(timer.stop())}, and the learning rate is {scheduler.get_last_lr()[0]}")
        print(f"the time for the Epoch: {epoch_index} is {timer_epoch.time_show(timer_epoch.stop())}")

    animator.save(f'{args.train_info}-loss-vs-epoch.png')
    # animator.pause()
    data.savemat(f'{args.train_info}.mat')


if __name__ == '__main__':
    args = parser.parse_args()
    setup_seed(args.seed)

    """ preparing the dataset """
    vocab = pickle.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab.token_to_idx
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    end_idx = token_to_idx["<END>"]
    # 训练与验证数据
    train_dataset = EurDataset(data_path=args.train_data_path)
    eval_dataset = EurDataset(data_path=args.test_data_path)
    print('训练数据集长度: {}'.format(len(train_dataset)))
    print('验证数据集长度: {}'.format(len(eval_dataset)))
    print(f'最大信道输出维度为{args.d_channel_max}')

    # DataLoader分割数据集为一个个batch
    collate_fn = collater(fixed_length_padding=args.fixed_padding, len_max=args.len_max + 2)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True, collate_fn=collate_fn, shuffle=True, drop_last=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True, collate_fn=collate_fn, shuffle=True, drop_last=False)

    # 在特定时间后开始运行程序
    time_waiting = 0  # 等待的秒数
    time.sleep(time_waiting)

    # 创建网络模型
    model = DeepSC(num_layers=args.num_layers, src_vocab_size=num_vocab, tgt_vocab_size=num_vocab, model_dim=args.d_model, channel_dim=args.d_channel_max,
                   num_heads=args.num_heads, dim_feedforward=args.dff, dropout=args.dropout).to(device)

    print("模型总参数:", sum(p.numel() for p in model.parameters()))

    resume = None
    timer = Timer()

    train_eval(model, train_dataloader, eval_dataloader, device=device, num_epoch=args.epochs,
               lr=args.lr, optim=args.optimizer, init=True, scheduler_type=args.scheduler_type,
               save_interval=30, eval_interval=1, save_path=args.checkpoint_path, resume=resume)
