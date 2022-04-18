#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from config import configurations
from train import train_process
from train import test_process
from train import inference

config = configurations()

def train():
    print('config : \n', vars(config))
    train_losses, val_losses, bleu_scores = train_process(config)
    # train_process -> model = build_model -> train(model)
    # model = Seq2Seq(Encoder, Deconder)
    #test_losses, bleu_scores = test_process(config)
    
def translate_seq2seq(source):
    source = source.rstrip()
    source = " ".join(source)
    result = inference(config, source)
    print(result)
    return " ".join(result)

if __name__ == '__main__':
    train()
