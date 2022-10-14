#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import sys

if __name__ == '__main__':
    if len(sys.argv) > 1:
        number = int(sys.argv[1])
    else:
        number = 1
    
    with open("./tmp/華", 'r') as f:
        華 = f.readlines()
    
    with open("./tmp/閩", 'r') as f:
        閩 = f.readlines()
        
    for _ in range(number):
        line = random.randint(0, len(華))
        print("line : {}\n".format(line))
        print("華 : {}".format(華[line]))
        print("閩 : {}".format(閩[line]))
        print("=========================")
