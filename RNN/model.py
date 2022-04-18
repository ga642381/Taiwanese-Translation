#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torch.utils.data.sampler as sampler
import torchvision
from torchvision import datasets, transforms

import numpy as np
import sys
import os
import random
import json


class Encoder(nn.Module):
    def __init__(self, 華_vocab_size, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(華_vocab_size, emb_dim)
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers,
                          dropout=dropout, batch_first=True, bidirectional=True)
        # (rnn): GRU(256, 1024, num_layers=3, batch_first=True, dropout=0.5)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        # input = [batch size, max_sequence len]
        """
        input[0] looks like this :
            torch.Size([60, 72])
            tensor([   1,  554,  711,  386, 1287,  133, 1511, 3049,    4,    2,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],
           device='cuda:0')

        """
        embedding = self.embedding(input)
        outputs, hidden = self.rnn(self.dropout(embedding))
        """
        outputs.shape :
            torch.Size([60, 72, 1024])
            
        hidden.shape:
            torch.Size([6, 60, 512])
        # 6 表示 n_layer(3) * 2 共 6 層， 最後一個 timestep 的 hidden state
        """
        # outputs = [batch size, sequence len, hid dim * directions]
        # hidden =  [num_layers * directions, batch size  , hid dim]
        # 幹太神奇了吧, batch_first=True 不會讓 hidden 的 batch_size 在最前面
        # outputs 是最上層RNN的輸出

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hid_dim):
        super(Attention, self).__init__()
        self.hid_dim = hid_dim

    def forward(self, encoder_outputs, decoder_hidden):
        # encoder_outputs = [batch size, sequence len, hid_dim * directions]
        # decoder_hidden = [num_layers, batch size, hid dim]
        # 一般來說是取 Encoder 最後一層的 hidden state 來做 attention
        """
            Decoder 的 forward : 
                attn = self.attention(encoder_outputs, hidden)

                hidden 為 Encoder最後 / Decoder 的 hidden state
            
            encoder_outputs.shape:
                torch.Size([60, 72, 1024])
                
            hidden.shape:
                torch.Size([3, 60, 1024])
        """ 
        decoder_hidden = decoder_hidden.permute(1, 2, 0)# (60, 1024, 3)
        matrix = torch.matmul(encoder_outputs, decoder_hidden) #(60, 72, 3)
        matrix = torch.mean(matrix, dim=2) #(60, 72)
        attention_weights = F.softmax(matrix ,dim=1) #(60, 72)
        attention = torch.matmul(encoder_outputs.transpose(1,2), attention_weights.unsqueeze(2)).transpose(1, 2)
        """
        print(attention.shape)
        torch.Size([60, 1, 1024])
        """
        #return attentino vector
        return attention


class Decoder(nn.Module):
    def __init__(self, 閩_vocab_size, emb_dim, hid_dim, n_layers, dropout, isatt):
        super().__init__()
        self.閩_vocab_size = 閩_vocab_size
        self.hid_dim = hid_dim * 2
        self.n_layers = n_layers
        self.embedding = nn.Embedding(閩_vocab_size, emb_dim)
        self.isatt = isatt
        self.attention = Attention(hid_dim)
        # 如果使用 Attention Mechanism 會使得輸入維度變化，請在這裡修改
        # e.g. Attention 接在輸入後面會使得維度變化，所以輸入維度改為
        self.input_dim = emb_dim + hid_dim * 2 if isatt else emb_dim
        #self.input_dim = emb_dim
        self.rnn = nn.GRU(self.input_dim, self.hid_dim,
                          self.n_layers, dropout=dropout, batch_first=True)
        self.embedding2vocab1 = nn.Linear(self.hid_dim, self.hid_dim * 2)
        self.embedding2vocab2 = nn.Linear(self.hid_dim * 2, self.hid_dim * 4)
        self.embedding2vocab3 = nn.Linear(self.hid_dim * 4, self.閩_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        # input = [batch size, vocab size]
        # hidden = [n_layer, batch_size, hid_dim*2] (modified by kaiwei)
        # Decoder 只會是單向，所以 directions=1
        """
        input.shape:
            torch.Size([60])
        hidden.shape:
            torch.Size([3, 60, 1024])
        """
        
        input = input.unsqueeze(1)
        
        """
        input.shape:
            torch.Size([60, 1])
        """
        embedded = self.dropout(self.embedding(input))
        # embedded = [batch size, 1, emb dim]
        """
        print(embedded.shape)
        embedded.shape:
            torch.Size([60, 1, 256])
        """
        if self.isatt:
            """
            print(encoder_outputs.shape)
            print(hidden.shape)
            
            encoder_outputs.shape:
                torch.Size([60, 72, 1024])
            hidden.shape:
                torch.Size([3, 60, 1024])
            """
            attn = self.attention(encoder_outputs, hidden)
            embedded = torch.cat((embedded, attn), dim=2)
            """
            print(embedded.shape)
            embedded.shape:
                torch.Size([60, 1, 1280])
            """
            # TODO: 在這裡決定如何使用 Attention，e.g. 相加 或是 接在後面， 請注意維度變化
        
        output, hidden = self.rnn(embedded, hidden)
        # output = [batch size, 1, hid dim]
        # hidden = [num_layers, batch size, hid dim]

        # 將 RNN 的輸出轉為每個詞出現的機率
        output = self.embedding2vocab1(output.squeeze(1))
        output = self.embedding2vocab2(output)
        prediction = self.embedding2vocab3(output)
        # prediction = [batch size, vocab size]
        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, input, target, teacher_forcing_ratio):
        # input  = [batch size, input len, vocab size]
        # target = [batch size, target len, vocab size]
        # teacher_forcing_ratio 是有多少機率使用正確答案來訓練
        batch_size = target.shape[0]
        target_len = target.shape[1]
        vocab_size = self.decoder.閩_vocab_size

        # 準備一個儲存空間來儲存輸出
        outputs = torch.zeros(batch_size, target_len,
                              vocab_size).to(self.device)
        # 將輸入放入 Encoder
        encoder_outputs, hidden = self.encoder(input)
        # Encoder 最後的隱藏層(hidden state) 用來初始化 Decoder
        # encoder_outputs 主要是使用在 Attention
        # 因為 Encoder 是雙向的RNN，所以需要將同一層兩個方向的 hidden state 接在一起
        # hidden =  [num_layers * directions, batch size  , hid dim]  --> [num_layers, directions, batch size  , hid dim]
        """
        hidden.shape:
            torch.Size([6, 60,512])
        """
        hidden = hidden.view(self.encoder.n_layers, 2, batch_size, -1)
        hidden = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=2)
        """
        hidden.shape:
            torch.Size([3, 60,1024])        
        """        
        
        # 取得 <BOS> token
        input = target[:, 0]
        preds = []
        for t in range(1, target_len):
            """
            input.shape:
                torch.Size([60])
            hidden.shape:
                torch.Size([3, 60, 1024])
            encoder_outputs.shape:
                torch.Size([60, 72, 1024])
            """
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[:, t] = output
            # 決定是否用正確答案來做訓練
            teacher_force = random.random() <= teacher_forcing_ratio
            # 取出機率最大的單詞
            top1 = output.argmax(1)
            # 如果是 teacher force 則用正解訓練，反之用自己預測的單詞做預測
            input = target[:, t] if teacher_force and t < target_len else top1
            preds.append(top1.unsqueeze(1))
        preds = torch.cat(preds, 1)
        return outputs, preds

    def inference(self, source):
        ########
        # TODO #
        ########
        # 在這裡實施 Beam Search
        # 此函式的 batch size = 1
        # source  = [batch size, source len, vocab size]
        # target = [batch size, target len, vocab size]
        batch_size = source.shape[0]
        source_len = source.shape[1]        # 取得最大字數
        vocab_size = self.decoder.閩_vocab_size

        # 準備一個儲存空間來儲存輸出
        outputs = torch.zeros(batch_size, source_len,
                              vocab_size).to(self.device)
        # 將輸入放入 Encoder
        encoder_outputs, hidden = self.encoder(source)
        # Encoder 最後的隱藏層(hidden state) 用來初始化 Decoder
        # encoder_outputs 主要是使用在 Attention
        # 因為 Encoder 是雙向的RNN，所以需要將同一層兩個方向的 hidden state 接在一起
        # hidden =  [num_layers * directions, batch size  , hid dim]  --> [num_layers, directions, batch size  , hid dim]
        hidden = hidden.view(self.encoder.n_layers, 2, batch_size, -1)
        hidden = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=2)
        
        
        # 取的 <BOS> token
        # input is just tensor([1], device='cuda:0')
        # input = target[:, 0] 
        d_input = source[:, 0]
        preds = []
        for t in range(1, source_len): #time step
            output, hidden = self.decoder(d_input, hidden, encoder_outputs)
            # 將預測結果存起來
            outputs[:, t] = output
            # 取出機率最大的單詞
            top1 = output.argmax(1)
            d_input = top1
            preds.append(top1.unsqueeze(1))
        preds = torch.cat(preds, 1)
        return outputs, preds

#=============================================================================#


def save_model(model, optimizer, store_model_path, step):
    torch.save(model.state_dict(), f'{store_model_path}/model_{step}.ckpt')
    return


def load_model(model, load_model_path):
    print(f'Load model from {load_model_path}')
    model.load_state_dict(torch.load(f'{load_model_path}'))
    return model


def build_model(config, 華_vocab_size, 閩_vocab_size, device):
    # 建構模型
    encoder = Encoder(華_vocab_size, config.emb_dim,
                      config.hid_dim, config.n_layers, config.dropout)
    decoder = Decoder(閩_vocab_size, config.emb_dim, config.hid_dim,
                      config.n_layers, config.dropout, config.attention)
    model = Seq2Seq(encoder, decoder, device)
    #print(model)
    # 建構 optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    #print(optimizer)
    if config.load_model:
        model = load_model(model, config.load_model_path)
    model = model.to(device)

    return model, optimizer
