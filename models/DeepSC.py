#!/usr/A/anaconda3/envs/py_3.8 python
# -*- coding: UTF-8 -*-
'''
@Project ：BiRC-DeepSC-EUR-8 
@File    ：DeepSC.py
@Author  ：Mingtong Zhang
@Date    ：2024/2/13 16:53 
'''
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Implement the PE function and dropout."""

    def __init__(self, model_dim, max_position_len=5000):
        super(PositionalEncoding, self).__init__()

        # 构造position embedding table
        pe_embedding_table = torch.zeros(max_position_len, model_dim)
        pos_col = torch.arange(0, max_position_len).reshape((-1, 1))  # [max_position_len, 1]
        pos_row = torch.pow(10000, torch.arange(0, model_dim, 2).reshape((1, -1)) / model_dim)  # [1, model_dim]
        pe_embedding_table[:, 0::2] = torch.sin(pos_col / pos_row)  # 对position矩阵的偶数列进行计算
        pe_embedding_table[:, 1::2] = torch.cos(pos_col / pos_row)  # 对position矩阵的奇数列进行计算
        # 利用nn.Embedding完成position embedding
        pe_embedding = nn.Embedding(max_position_len, model_dim)
        pe_embedding.weight = nn.Parameter(pe_embedding_table, requires_grad=False)
        pos = torch.arange(max_position_len).unsqueeze(0).to(torch.int32)
        pe_embedding = pe_embedding(pos)  # [1, max_position_len, model_dim]

        self.register_buffer('pe', pe_embedding)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, model_dim):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert model_dim % num_heads == 0
        # We assume d_v always equals d_k
        self.d_k = model_dim // num_heads
        self.num_heads = num_heads

        self.wq = nn.Linear(model_dim, model_dim)
        self.wk = nn.Linear(model_dim, model_dim)
        self.wv = nn.Linear(model_dim, model_dim)

        self.out_dense = nn.Linear(model_dim, model_dim)
        self.attn = None

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query = self.wq(query).view(nbatches, -1, self.num_heads, self.d_k)
        query = query.transpose(1, 2)
        key = self.wk(key).view(nbatches, -1, self.num_heads, self.d_k)
        key = key.transpose(1, 2)
        value = self.wv(value).view(nbatches, -1, self.num_heads, self.d_k)
        value = value.transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(query, key, value, mask=mask)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.num_heads * self.d_k)

        x = self.out_dense(x)
        return x

    def attention(self, query, key, value, mask=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        # print(mask.shape)
        if mask is not None:
            # 根据mask，指定位置填充 -1e9
            scores = scores + (mask * -1e9)
            # attention weights
        p_attn = F.softmax(scores, dim=-1)
        return torch.matmul(p_attn, value), p_attn


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, model_dim, dim_feedforward, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(model_dim, dim_feedforward)
        self.w_2 = nn.Linear(dim_feedforward, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.w_2(x)
        return x


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class SubLayerConnection(nn.Module):

    def __init__(self, model_dim, dropout=0.1):
        super(SubLayerConnection, self).__init__()
        self.layernorm = nn.LayerNorm(model_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.layernorm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, dim_feedforward, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.self_mha = MultiHeadedAttention(num_heads, model_dim)
        self.ffn = PositionwiseFeedForward(model_dim, dim_feedforward, dropout=dropout)
        self.sublayer = clones(SubLayerConnection(model_dim, dropout), 2)

    def forward(self, x, src_key_padding_mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_mha(x, x, x, src_key_padding_mask))
        return self.sublayer[1](x, self.ffn)


class Encoder(nn.Module):
    def __init__(self, num_layers, src_vocab_size, model_dim, num_heads, dim_feedforward, dropout=0.1):
        super(Encoder, self).__init__()

        self.d_model = model_dim
        self.num_heads = num_heads
        self.embedding = nn.Embedding(src_vocab_size, model_dim)
        self.pos_encoding = PositionalEncoding(model_dim)
        self.enc_layers = nn.ModuleList([EncoderLayer(model_dim, num_heads, dim_feedforward, dropout) for _ in range(num_layers)])

    def forward(self, x, src_key_mask=None):
        # the input size of x is [batch_size, src_seq_len]
        # the shape of src_key_mask is [batch_size, 1, src_seq_len]
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        for enc_layer in self.enc_layers:
            x = enc_layer(x, src_key_padding_mask=src_key_mask)
        return x


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward"

    def __init__(self, model_dim, num_heads, dim_feedforward, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_mha = MultiHeadedAttention(num_heads, model_dim)
        self.mutual_mha = MultiHeadedAttention(num_heads, model_dim)
        self.ffn = PositionwiseFeedForward(model_dim, dim_feedforward, dropout=dropout)
        self.sublayer = clones(SubLayerConnection(model_dim, dropout), 3)

    def forward(self, x, memory, look_ahead_mask, tgt_key_padding_mask, memory_key_padding_mask):
        if look_ahead_mask is not None:
            look_ahead_mask = look_ahead_mask.expand(x.shape[0], look_ahead_mask.shape[0], look_ahead_mask.shape[1])
            combined_mask = torch.max(look_ahead_mask, tgt_key_padding_mask)
        else:
            combined_mask = tgt_key_padding_mask

        x = self.sublayer[0](x, lambda x: self.self_mha(x, x, x, combined_mask))
        x = self.sublayer[1](x, lambda x: self.mutual_mha(x, memory, memory, memory_key_padding_mask))
        return self.sublayer[2](x, self.ffn)


class Decoder(nn.Module):
    def __init__(self, num_layers, trg_vocab_size,
                 model_dim, num_heads, dim_feedforward, dropout=0.1):
        super(Decoder, self).__init__()

        self.d_model = model_dim
        self.num_heads = num_heads
        self.embedding = nn.Embedding(trg_vocab_size, model_dim)
        self.pos_encoding = PositionalEncoding(model_dim)
        self.dec_layers = nn.ModuleList([DecoderLayer(model_dim, num_heads, dim_feedforward, dropout) for _ in range(num_layers)])

    def forward(self, x, memory, look_ahead_mask=None, tgt_key_mask=None, memory_key_mask=None):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        for dec_layer in self.dec_layers:
            x = dec_layer(x, memory, look_ahead_mask=look_ahead_mask, tgt_key_padding_mask=tgt_key_mask,
                          memory_key_padding_mask=memory_key_mask)
        return x


class JSC_Encoder(nn.Module):
    def __init__(self, in_features_dim, intermediate_dim, out_features_dim):
        super(JSC_Encoder, self).__init__()

        self.linear1 = nn.Linear(in_features_dim, intermediate_dim)
        self.linear2 = nn.Linear(intermediate_dim, out_features_dim)

    def forward(self, x):
        x1 = self.linear1(x)
        x = F.relu(x1)
        output = self.linear2(x)

        return output


class JSC_Decoder(nn.Module):
    def __init__(self, in_features_dim, intermediate_dim, out_features_dim):
        super(JSC_Decoder, self).__init__()

        self.linear1 = nn.Linear(in_features_dim, out_features_dim)
        self.linear2 = nn.Linear(out_features_dim, intermediate_dim)
        self.linear3 = nn.Linear(intermediate_dim, out_features_dim)
        self.layernorm = nn.LayerNorm(out_features_dim, eps=1e-6)

    def forward(self, x):
        x1 = self.linear1(x)
        x = F.relu(self.linear2(x1))
        x = self.linear3(x)
        output = self.layernorm(x1 + x)

        return output


class DeepSC(nn.Module):
    def __init__(self, num_layers, src_vocab_size, tgt_vocab_size, model_dim, channel_dim, num_heads, dim_feedforward, dropout=0.1):
        super(DeepSC, self).__init__()

        self.semantic_encoder = Encoder(num_layers, src_vocab_size, model_dim, num_heads, dim_feedforward, dropout)

        self.jsc_encoder = JSC_Encoder(in_features_dim=model_dim, intermediate_dim=2 * model_dim, out_features_dim=channel_dim)

        self.jsc_decoder = JSC_Decoder(in_features_dim=channel_dim, intermediate_dim=2 * model_dim, out_features_dim=model_dim)

        self.semantic_decoder = Decoder(num_layers, tgt_vocab_size, model_dim, num_heads, dim_feedforward, dropout)

        self.dense = nn.Linear(model_dim, tgt_vocab_size)
