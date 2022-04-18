#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.utils.data as data
import torch
import numpy as np
import os
import json
import re


class LabelTransform(object):
    def __init__(self, size, pad):
        self.size = size
        self.pad = pad

    def __call__(self, label):
        label = np.pad(
            label, (0, (self.size - label.shape[0])), mode='constant', constant_values=self.pad)
        return label


class 華閩Dataset(data.Dataset):
    def __init__(self, root, max_output_len, mode):
        self.root = root
        self.mode = mode
        self.word2int_華, self.int2word_華 = self.get_dictionary('華')
        self.word2int_閩, self.int2word_閩 = self.get_dictionary('閩')

        # 載入資料
        self.data = []
        if mode == 'deploy':
            pass
        
        else :
            with open(os.path.join(self.root, f'{mode}.txt'), "r") as f:
                for line in f:
                    self.data.append(line)
            print(f'{mode} dataset size: {len(self.data)}')
            

        self.華_vocab_size = len(self.word2int_華)
        self.閩_vocab_size = len(self.word2int_閩)
        self.transform = LabelTransform(max_output_len, self.word2int_華['<PAD>'])
        
    def replace_data(self, source:list):
        self.data = source

    def get_dictionary(self, language):
        # 載入字典
        with open(os.path.join(self.root, f'word2int_{language}.json'), "r") as f:
            word2int = json.load(f)
        with open(os.path.join(self.root, f'int2word_{language}.json'), "r") as f:
            int2word = json.load(f)
        return word2int, int2word
    
    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, Index):
        # 預備特殊字元
        BOS = self.word2int_華['<BOS>']
        EOS = self.word2int_華['<EOS>']
        UNK = self.word2int_華['<UNK>']
        
        if self.mode != "deploy":
            # 先將中英文分開
            sentences = self.data[Index]
            sentences = re.split('[\t\n]', sentences)
            sentences = list(filter(None, sentences))
            #print (sentences)
            assert len(sentences) == 2
            
        else:
            sentence = self.data[Index] # one sentence
        

        # 在開頭添加 <BOS>，在結尾添加 <EOS> ，不在字典的 subword (詞) 用 <UNK> 取代
        華, 閩 = [BOS], [BOS]
        if self.mode != "deploy":
            # 將句子拆解為 subword 並轉為整數
            sentence = re.split(' ', sentences[0])
            sentence = list(filter(None, sentence))
            #print (f'en: {sentence}')
            for word in sentence:
                華.append(self.word2int_華.get(word, UNK))
            華.append(EOS)
            # 將句子拆解為單詞並轉為整數
            # e.g. < BOS >, we, are, friends, < EOS > --> 1, 28, 29, 205, 2
            sentence = re.split(' ', sentences[1])
            sentence = list(filter(None, sentence))
            #print (f'cn: {sentence}')
            for word in sentence:
                閩.append(self.word2int_閩.get(word, UNK))
            閩.append(EOS)
        
        else:
            sentence = re.split(' ', sentence)
            sentence = list(filter(None, sentence))
            for word in sentence:
                華.append(self.word2int_華.get(word, UNK))
            華.append(EOS)
            閩.append(EOS)

        華, 閩 = np.asarray(華), np.asarray(閩)
        # 用 <PAD> 將句子補到相同長度
        華, 閩 = self.transform(華), self.transform(閩)
        華, 閩 = torch.LongTensor(華), torch.LongTensor(閩)
        return 華, 閩
