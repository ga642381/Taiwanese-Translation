#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
icorpus_華 = './icorpus/華' 
icorpus_閩 = './icorpus/閩_臺羅'
out_華 = './tmp/icorpus_華' 
out_閩 = './tmp/icorpus_閩'

with open(icorpus_華, "r") as f:
    華 = f.readlines()

with open(icorpus_閩, "r") as f:
    閩 = f.readlines()

eng_num_words = []
to_remove_index = []
##=============== 華 ===============##
for i, l in enumerate(華):
    l = l.replace("-", " ")
    if re.match(r'[\u4e00-\u9fff]', l[-2]):
        l = l.replace("\n", " 。\n")
    
    # ============================= #
    for token in l.split():
        # english words
        if re.findall(r'[a-zA-z0-9]', token):
            chinese = re.findall(r'[\u4e00-\u9fff]', token)
            if chinese:
                for c in chinese:
                    token = token.replace(c, "")
            eng_num_words.append((i, token))
    # ============================= #
    
    # === to char === #
    l_ = []
    for j, c in enumerate(l):
        l_.append(c)
        
        if j == len(l) -1:
            break        
        
        if l[j+1] != " " and c!= " ":
            l_.append(" ")
    # ============== #
    
    l_ = "".join(l_)
    l_ = l_.rstrip()
    
    華[i] = l_
##=============== 華 ===============##
##=============== 閩 ===============##
for i, l in enumerate(閩):
    l = l.replace("「", "\"")
    l = l.replace("」", "\"")
    l = l.replace("『", "\'")
    l = l.replace("』", "\'")

    
    l = l.replace("，", ",")
    l = l.replace("。", ".")
    l = l.replace("？", "?")    
    l = l.replace("「", "\"")
    l = l.replace("」", "\"")
    l = l.replace("；", ";")
    l = l.replace("、", ",")
    l = l.replace("！", "!")
    l = l.replace("：", ":")
    
    if re.match(r'[0-9a-zA-z]', l[-2]):
        l= l.replace("\n", " .\n")
    
    l = l.replace("-", " - ")
    l = l.rstrip()
    
    閩[i] = l
    
for line, eng_num in eng_num_words:
    if eng_num in 閩[line]:
        閩[line] = 閩[line].replace(eng_num, " ".join(eng_num))


for i, line in enumerate(閩):
    tokens = line.split()
    for token in tokens:
        if not (re.findall(r'[a-zA-Z]', token) and re.findall(r'[0-9]', token)):
            if len(token)!=1:
                to_remove_index.append(i)
                
        if len(token) > 1:
            if not re.findall(r'[a-z]', token[0]):
                to_remove_index.append(i)
                
閩 =  [j for i, j in enumerate(閩) if i not in to_remove_index]
華 =  [j for i, j in enumerate(華) if i not in to_remove_index]
assert( len(閩) == len(華))

##=============== 閩 ===============##
with open(out_華, "w") as f:
    for line in 華:
        f.write(line)
        f.write("\n")
    print(f.name)   

with open(out_閩, "w") as f:
    for line in 閩:
        f.write(line)
        f.write("\n")
    print(f.name)      

            