# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 21:47:18 2017


"""
import sys
from tqdm import tqdm
import re


if len(sys.argv)<=2:
    print('Недостаточно параметров, должно быть два файла входящий и обработанный')
    raise SystemExit(2)


def get_file(arg):
    file=sys.argv[arg]
    if len(file)==0:
        print('Не указан файл',arg)
        raise SystemExit(1)
    return file

file_name_input=get_file(1)
file_name_output=get_file(2)

#tokens={'javascript':1, 'java':2, 'python':3, 'ruby':4, 'php':5, 'c++':6, 'c#':7, 'go':8, 'scala':9, 'swift':10}
reg=re.compile('(javascript)|(java)|(python)|(ruby)|(php)|(c\+\+)|(c\#)|(go)|(scala)|(swift)')
table = str.maketrans(':|', '  ')

f = open(file_name_input,'r')
f_out = open(file_name_output,'w')
pbar = tqdm()
strnum=0
for line_str in f:
    pbar.update(1)
#==============================================================================
#     скрипт должен работать с аргументами командной строки: с путями к файлам на входе и на выходе
#     строки обрабатываются по одной (можно использовать tqdm для подсчета числа итераций)
#     если табуляций в строке нет или их больше одной, считаем строку поврежденной и пропускаем
#     в противном случае смотрим, сколько в строке тегов из списка javascript, java, python, ruby, php, c++, c#, go, scala и swift.
#        Если ровно один, то записываем строку в выходной файл в формате VW: label | text, где label – число от 1 до 10 (1 - javascript, ... 10 – swift).
        #Пропускаем те строки, где интересующих тегов больше или меньше одного
#     из текста вопроса надо выкинуть двоеточия и вертикальные палки, если они есть – в VW это спецсимволы
#==============================================================================
    if line_str.count('\t')!=1: continue#если табуляций в строке нет или их больше одной, считаем строку поврежденной и пропускаем
    ind=line_str.find('\t')

    m=reg.findall(line_str[ind+1:])
    if len(m)!=1: continue#Пропускаем те строки, где интересующих тегов больше или меньше одного    
    for token_index in filter(lambda val: len(val[1])>0, enumerate(m[0],1)):#получаем индекс токена       
        textq=line_str[:ind-1].translate(table).lstrip()
        f_out.write(str(token_index[0,0]) +'|' + textq + '\n')
        strnum+=1
        break

pbar.close()
print('Количество записей: ',strnum)
print('Должно быть       :  4389054')
