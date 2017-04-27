# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 22:12:45 2017
"""
from tqdm import tqdm
import re

f_in=open('stackoverflow.vw','r')
f_train = open('stackoverflow_train.vw','w')#начало
f_test = open('stackoverflow_test.vw','w')#конец
f_valid = open('stackoverflow_valid.vw','w')#середина

f_valid_labels=open('stackoverflow_valid_labels.txt','w')
f_test_labels=open('stackoverflow_test_labels.txt','w')

total_strs=sum(1 for l in f_in)
print('lines: ',total_strs)

f_in.seek(0)
split1=1463018
split2=total_strs-1463018

f_trainl=0
f_testl=0
f_validl=0

reg=re.compile('(\d+)\s\|')
def save_target(file,_str):
    res=reg.match(_str)
    file.write(res.group(1)+'\n')


strnum=0
pbar = tqdm()
for line_str in f_in:
    pbar.update(1)

    if strnum<split1:
        f_train.write(line_str)
        f_trainl+=1
    elif strnum>=split2:
        f_test.write(line_str)
        f_testl+=1
        save_target(f_test_labels,line_str)
    else:
        f_valid.write(line_str)
        f_validl+=1
        save_target(f_valid_labels,line_str)

    strnum+=1

pbar.close()


print('f_trainl=\t',f_trainl)
print('f_testl=\t',f_testl)
print('f_validl=\t',f_validl)

f_in.close()
f_train.close()
f_test.close()
f_valid.close()
f_valid_labels.close()
f_test_labels.close()