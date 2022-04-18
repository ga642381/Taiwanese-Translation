#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
TGB_華 = "./TGB/TGB_華語"
TGB_閩 = "./TGB/TGB_閩南語"
out_華 = "./tmp/TGB_華"
out_閩 = "./tmp/TGB_閩"

華_ = []
閩_ = []
to_remove_lines = []
to_remove_index = []
eng_num_words = []

SYMBOLS = ['~', "'", '+', '[',
           '\\', '@', '^', '{',
            '-', '"', '*', '|',
            '&', '<', '`', '}',
            '_', '=', ']', '>',
            '#', '$', '/', '...',
            '』', '『','／']
            
with open(TGB_華, "r") as f:
    華 = f.readlines()

with open(TGB_閩, "r") as f:
    閩 = f.readlines()

def to_pingin(閩):
    閩_tmp = []
    for i, l in enumerate(閩):
        tokens = l.split()
        join_elms = []
        for t in tokens:
            if "｜" in t:
                token = t.split("｜") # len == 2
                left = token[0] # e5-人
                right = token[-1] # -lang5
                
                left_tmp = re.findall(r'[\u2E80-\u9fff]+',left) # find中文字
                right_tmp = right.split("-") #找取代的拼音
                while "" in right_tmp:
                    right_tmp.remove("")
                
                # 不知為何，會有0lang 0tit4 這種情況發生
                for i_, pinyin in enumerate(right_tmp):
                    # 躡(nih4) -> 內(lai7)
                    if pinyin=="nih4":
                        right_tmp[i_] = "lai7"
                        
                    if pinyin[0].isdigit():
                        right_tmp[i_] = pinyin[1:]
                        
                # 如果左邊文字不太對勁，右邊又不是標點符號而是拼音，直接取右邊。
                if len(left_tmp) != len(right_tmp) and re.match(r'[0-9a-zA-Z]',right):
                    left = right
                else:
                    for i_, chi in enumerate(left_tmp):
                        left = left.replace(chi, right_tmp[i_])
                        
                join_elms.append(left)
                
            else:
                if re.findall(r'[\u2E80-\u9fff]+', t):
                    to_remove_lines.append(l)
                    to_remove_index.append(i)
                    #print(i, t)
                else:
                    #print(t)
                    join_elms.append(t)
                
        l = " ".join(join_elms)
        # char:
        l = l.replace("-", " - ")
        l = l + "\n"
        閩_tmp.append(l)
    return 閩_tmp

閩_tmp = to_pingin(閩)
##=============== 華 ===============##
for i, l in enumerate(華) :
    for s in SYMBOLS:
        if s in l:
           to_remove_lines.append(l) 
           to_remove_index.append(i)
           
    if "《 TGB 通訊 》" in l:
        to_remove_lines.append(l)
        to_remove_index.append(i)     
        
    if len(l) <= 2:
        to_remove_lines.append(l)
        to_remove_index.append(i)        
        
    l = l.replace("“","「")
    l = l.replace("”","」")
    l = l.replace("@", "")
    
    if ("「 " in l) and ("」 " not in l):
        l = l.replace("「 ", "")
    
    if  ("」 " in l) and ("「 " not in l):
        l = l.replace("」 ", "")     
        
    if  ("」 " in l) and ("「 " in l):
        if l.index("」") < l.index("「"):
            l = l.replace("「 ", "")
            l = l.replace("」 ", "")
    
    
    # === find english and numbers === #
    for token in l.split():
        # english words
        if re.findall(r'[a-zA-z0-9]', token):
            chinese = re.findall(r'[\u4e00-\u9fff]', token)
            if chinese:
                for c in chinese:
                    token = token.replace(c, "")
            eng_num_words.append((i, token))
            
    # ================================ #
    
    l_ = []
    for i, c in enumerate(l):
        l_.append(c)
        
        if i == len(l) -1:
            break        
        
        if l[i+1] != " " and c!= " ":
            l_.append(" ")
    l_ = "".join(l_)
    l_ = l_.rstrip()
    華_.append(l_)
##=============== 華 ===============##
##=============== 閩 ===============##
for i, l in enumerate(閩_tmp) :
    l = l.replace("“","「")
    l = l.replace("”","」")    

    if ("「 " in l) and ("」 " not in l):
        l = l.replace("「 ", "")
    
    if  ("」 " in l) and ("「 " not in l):
        l = l.replace("」 ", "")     
        
    if  ("」 " in l) and ("「 " in l):
        if l.index("」") < l.index("「"):
            l = l.replace("「 ", "")
            l = l.replace("」 ", "")
            
    l = l.replace("，", ",")
    l = l.replace("。", ".")
    l = l.replace("？", "?")    
    l = l.replace("「", "\"")
    l = l.replace("」", "\"")
    l = l.replace("；", ";")
    l = l.replace("、", ",")
    l = l.replace("！", "!")
    l = l.replace("：", ":")
    l = l.replace("『", "\'")
    l = l.replace("』", "\'")
    l = l.replace("@", "")
    l = l.rstrip()
    
    #if not l[0].isalpha():
    #    print(i, l)
    閩_.append(l)
# === replace english number words === #
for line, eng_num in eng_num_words:
    if eng_num in 閩_[line]:
        閩_[line] = 閩_[line].replace(eng_num, " ".join(eng_num))

for i, line in enumerate(閩_):
    tokens = line.split()
    for token in tokens:
        if not (re.findall(r'[a-zA-Z]', token) and re.findall(r'[0-9]', token)):
            if len(token)!=1:
                to_remove_index.append(i)
        
        if len(token) > 1:
            if not re.findall(r'[a-z]', token[0]):
                to_remove_index.append(i)

閩_ =  [j for i, j in enumerate(閩_) if i not in to_remove_index]
華_ =  [j for i, j in enumerate(華_) if i not in to_remove_index]

assert( len(閩_) == len(華_))


with open(out_華, "w") as f:
    for line in 華_:
        f.write(line)
        f.write("\n")
    print(f.name)
    
with open(out_閩, "w") as f:
    for line in 閩_:
        f.write(line)
        f.write("\n")
    print(f.name)        

            
            
            