'''
@Project ：BiMSC 
@File    ：FD-DeepSC-bleu.py
@Author  ：Mingtong Zhang
@Date    ：2024/11/11 16:41 
'''
import pickle
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import EurDataset, collater
from models.DeepSC import DeepSC
from utils.utils_general import setup_seed, Timer, Data_savemat, BleuScore, SeqtoText
from utils.utils_for_FD_DeepSC import greedy_decode
from config.test_config import get_test_parser

# Some global simulation parameters for debugging
channel = 'AWGN'
trained_epoch = None
best_mode = True
train_config_number = 0
test_config_number = 0
test_rate_list = [2, 4, 6, 8, 10, 12, 14, 16]

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

parser = get_test_parser(case='FD-DeepSC', channel=channel, train_config_number=train_config_number, test_config_number=test_config_number,
                         trained_epoch=trained_epoch, best_mode=best_mode, test_rate_list=test_rate_list)

np.set_printoptions(precision=4)
torch.set_printoptions(precision=4)


def performance(args, model, test_dataloader, SNR, rate):
    print('testing on:', device)
    bleu_score_1gram = BleuScore(1, 0, 0, 0)
    bleu_score_2gram = BleuScore(0, 1, 0, 0)
    bleu_score_3gram = BleuScore(0, 0, 1, 0)
    bleu_score_4gram = BleuScore(0, 0, 0, 1)

    S2T = SeqtoText(token_to_idx, end_idx)
    score1, score2, score3, score4 = [], [], [], []

    model.eval()
    with torch.no_grad():
        for loop in range(args.test_loop):
            Tx_word = []
            Rx_word = []
            timer.start()

            for snr in SNR:
                print(f'-----------SNR = {snr}: 测试开始------------')
                timer_SNR = Timer()
                timer_SNR.start()
                predicted_word = []
                target_word = []

                test_bar = tqdm(test_dataloader)

                for sents_test in test_bar:
                    sents_test = sents_test.to(device)  # 将一个batch的测试数据送入CPU or GPU
                    snr_extend = snr * torch.ones(sents_test.size(0)).to(device)

                    out = greedy_decode(model=model, sent=sents_test, pad=pad_idx, end=end_idx, snr=snr_extend,
                                        len_max=args.len_max, start_symbol=start_idx, channel=args.channel, rate=rate,
                                        device=device)

                    sentences = out.tolist()
                    result_string = list(map(S2T.sequence_to_text, sentences))
                    predicted_word = predicted_word + result_string

                    target_sent = sents_test.tolist()
                    result_string = list(map(S2T.sequence_to_text, target_sent))
                    target_word = target_word + result_string

                    test_bar.set_description(f"SNR: {snr} dB")

                Rx_word.append(predicted_word)
                Tx_word.append(target_word)

                print(f"the time for the SNR: {snr} dB is {timer_SNR.time_show(timer_SNR.stop())}")
            print(f"the test for greedy decode has taken {timer.time_show(timer.stop())}")

            timer_bleu = Timer()
            timer_bleu.start()
            bleu1_score = []
            bleu2_score = []
            bleu3_score = []
            bleu4_score = []
            for sent_Rx, sent_Tx in zip(Rx_word, Tx_word):
                bleu1_score.append(bleu_score_1gram.compute_blue_score(sent_Rx, sent_Tx, handcraft_mode=True))
                bleu2_score.append(bleu_score_2gram.compute_blue_score(sent_Rx, sent_Tx, handcraft_mode=True))
                bleu3_score.append(bleu_score_3gram.compute_blue_score(sent_Rx, sent_Tx, handcraft_mode=True))
                bleu4_score.append(bleu_score_4gram.compute_blue_score(sent_Rx, sent_Tx, handcraft_mode=True))

            score1.append(np.mean(np.array(bleu1_score), axis=1))
            score2.append(np.mean(np.array(bleu2_score), axis=1))
            score3.append(np.mean(np.array(bleu3_score), axis=1))
            score4.append(np.mean(np.array(bleu4_score), axis=1))

            print(f"Test Loop: {loop + 1} is end")
            print(f"the test for bleu calculation has taken {timer_bleu.time_show(timer_bleu.stop())}")
            print(f"the test has taken {timer.time_show(timer.stop())}")

        score1 = np.mean(np.array(score1), axis=0)
        score2 = np.mean(np.array(score2), axis=0)
        score3 = np.mean(np.array(score3), axis=0)
        score4 = np.mean(np.array(score4), axis=0)
        bleu_score = 0.25 * score1 + 0.25 * score2 + 0.25 * score3 + 0.25 * score4

        return score1, score2, score3, score4, bleu_score


if __name__ == '__main__':
    args = parser.parse_args()
    setup_seed(args.seed)
    SNR = [-3, 0, 3, 5, 6, 9, 12, 15, 18]
    """ preparing the dataset """
    vocab = pickle.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab.token_to_idx
    idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]

    test_dataset = EurDataset(data_path=args.test_data_path)
    collate_fn = collater(fixed_length_padding=args.fixed_padding, len_max=args.len_max + 2)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_bs, num_workers=0, pin_memory=True, collate_fn=collate_fn, shuffle=True)

    # 在特定时间后开始运行程序
    time_waiting = 0  # 等待的秒数
    time.sleep(time_waiting)

    model = DeepSC(num_layers=args.num_layers, src_vocab_size=num_vocab, tgt_vocab_size=num_vocab, model_dim=args.d_model, channel_dim=args.d_channel_max,
                   num_heads=args.num_heads, dim_feedforward=args.dff, dropout=args.dropout).to(device)

    model_path = args.trained_path
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"loading from {model_path}")

    timer = Timer()
    for index, test_rate in enumerate(args.test_rate_list):
        bleu_score = performance(args, model, test_dataloader, SNR, test_rate)
        print(bleu_score)

        data = Data_savemat()
        for i in range(len(bleu_score[0])):
            data.append(x=SNR[i], y=(bleu_score[0][i], bleu_score[1][i], bleu_score[2][i], bleu_score[3][i], bleu_score[4][i]))
        data.savemat(filename=f'results/{args.test_info_list[index]}-BLEU.mat')
