#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import random
tmp_dir = './tmp'
template = '{}_{}'
data_types = ['icorpus', 'TGB']

out_train = '../data/training.txt'
out_test = '../data/testing.txt'
out_val = '../data/validation.txt'

# ===================================== #
華 = []
閩 = []
for d in data_types:
    file = tmp_dir + "/" + template.format(d, '華')
    with open(file, 'r') as f:
        華 +=  f.readlines()
    
    file = tmp_dir + "/" + template.format(d, '閩')
    with open(file, 'r') as f:
        閩 +=  f.readlines()

# ====================== 處理太長的句子 ==================#
too_long_index = []
for i, line in enumerate(華):
    if len(line.split()) > 95:
        too_long_index.append(i)
        
for i, line in enumerate(閩):
    if len(line.split()) > 95:
        too_long_index.append(i)

閩 =  [j for i, j in enumerate(閩) if i not in too_long_index]
華 =  [j for i, j in enumerate(華) if i not in too_long_index]

out_華 = tmp_dir + '/華'
out_閩 = tmp_dir + '/閩'
with open(out_華, "w") as f:
    for line in 華:
        f.write(line)

with open(out_閩, "w") as f:
    for line in 閩:
        f.write(line)

    
# =============== Split ================ #
華閩 = []
for index in range(len(華)):
    l = 華[index] + ' \t' + 閩[index]
    l = l.replace('\n', '')
    l = l + " \n"
    華閩.append(l)

SEED = 40666888
random.seed(SEED)
random.shuffle(華閩)

# total 110995
train_len = 110000
test_len = 495
val_len = 500

train_華閩 = 華閩[:train_len]
test_華閩 = 華閩[train_len:-val_len]
val_華閩 = 華閩[-val_len:]

print("len train :{}".format(len(train_華閩)))
print("len test  :{}".format(len(test_華閩 )))
print("len val   :{}".format(len(val_華閩  )))


with open(out_train, "w") as f:
    print(out_train)
    f.writelines(train_華閩)

with open(out_test, "w") as f:
    print(out_test)
    f.writelines(test_華閩)
    
with open(out_val, "w") as f:
    print(out_val)
    f.writelines(val_華閩)

























