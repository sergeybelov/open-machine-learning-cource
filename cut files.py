# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 22:12:45 2017


"""
from tqdm import tqdm


f_in=open('stackoverflow.vw','r')
f_train = open('stackoverflow_train.vw','w')#начало
f_test = open('stackoverflow_test.vw','w')#конец
f_valid = open('stackoverflow_valid.vw','w')#середина

total_strs=sum(1 for l in f_in)
print('lines: ',total_strs)

f_in.seek(0)
split1=1463018
split2=total_strs-1463018

f_trainl=0
f_testl=0
f_validl=0


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
    else:
        f_valid.write(line_str)
        f_validl+=1

    strnum+=1

pbar.close()


print('f_trainl=\t',f_trainl)
print('f_testl=\t',f_testl)
print('f_validl=\t',f_validl)
