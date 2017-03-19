# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 13:34:33 2017
 

"""
from __future__ import division, print_function
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (12, 10)
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from math import log

#==============================================================================
# Создание набора данных
#==============================================================================
# Создание датафрейма с dummy variables
def create_df(dic, feature_list):
    out = pd.DataFrame(dic)
    out = pd.concat([out, pd.get_dummies(out[feature_list])], axis = 1)
    out.drop(feature_list, axis = 1, inplace = True)
    return out

# Некоторые значения признаков есть в тесте, но нет в трейне и наоборот
def intersect_features(train, test):
    common_feat = list( set(train.keys()) & set(test.keys()))
    return train[common_feat], test[common_feat]


features = ['Внешность', 'Алкоголь_в_напитке','Уровень_красноречия','Потраченные_деньги']
#Обучающая выборка
df_train = {}
df_train['Внешность'] = ['приятная','приятная','приятная','отталкивающая','отталкивающая','отталкивающая','приятная']
df_train['Алкоголь_в_напитке'] = ['да','да','нет','нет','да','да','да']
df_train['Уровень_красноречия'] = ['высокий','низкий','средний','средний','низкий','высокий','средний']
df_train['Потраченные_деньги'] = ['много','мало','много','мало','много','много','много']
df_train['Поедет'] = LabelEncoder().fit_transform(['+','-','+','-','-','+','+'])
df_train = create_df(df_train, features)

#Тестовая выборка
df_test = {}
df_test['Внешность'] = ['приятная','приятная','отталкивающая']
df_test['Алкоголь_в_напитке'] = ['нет','да','да']
df_test['Уровень_красноречия'] = ['средний','высокий','средний']
df_test['Потраченные_деньги'] = ['много','мало','много']
df_test = create_df(df_test, features)


# Некоторые значения признаков есть в тесте, но нет в трейне и наоборот
y = df_train['Поедет']
df_train, df_test = intersect_features(train = df_train, test = df_test)

#Какова энтропия начальной системы ($S_0$)?
#Под состояниями системы понимаем значения признака "Поедет" – 0 или 1 (то есть всего 2 состояния).
#==============================================================================
# Какова энтропия начальной системы (S0)? Под состояниями системы понимаем значения признака "Поедет" – 0 или 1
# в обучающей выборке (то есть всего 2 состояния).
#==============================================================================
def ShennonEntropy(array,vals):
    sum=0.
    count=array.count()
    for val in vals:
        temp=float(array[array==val].count())/count
        if temp==0: continue
        sum-=temp*log(temp,2)
    return round(sum,3)

s0=ShennonEntropy(y,[0,1])
print("S0=%f" % (s0))
print("")

#==============================================================================
# Вопрос 3.2. Рассмотрим разбиение обучающей выборки по признаку "Внешность_приятная".
# Какова энтропия S1 левой группы, тех, у кого внешность приятная, и правой группы – S2 ?
# Каков прирост информации при данном разбиении (IG)? Отметьте все верные ответы.
#
#==============================================================================
#df_train.sort_values('Внешность_приятная', ascending=[True],inplace=True)#сначала сортируем по возрастанию

#S1 = 0,967
#S2 = 0,918 *
#IG = 0,128 *

#S1 = 0,811 *
#S2 = 0,826
#IG = 0,178
#ищем индекс при котором меняются значения
for i in range(df_train['Внешность_приятная'].count()-1):
    #print("i=%i" % (i+1))
    count=df_train['Внешность_приятная'].count()
    s1=ShennonEntropy(df_train.loc[:i,'Внешность_приятная'],[0,1])
    s2=ShennonEntropy(df_train.loc[i+1:,'Внешность_приятная'],[0,1])
    iqg=round(s0-(i+1)/count*s1-(count-i-1)/count*s2,3)

    if s1 in [0.967,0.811]: print("S1=%f" % s1)
    if s2 in [0.918,0.826]: print("S2=%f" % s2)
    if iqg in [0.128,0.178]: print("iqQ=%f" % iqg)


#Часть 2. Функции для расчета энтропии и прироста информации
print("")

balls = [1 for i in range(9)] + [0 for i in range(11)]

# две группы
balls_left  = [1 for i in range(8)] + [0 for i in range(5)] # 8 синих и 5 желтых
balls_right = [1 for i in range(1)] + [0 for i in range(6)] # 1 синий и 6 желтых


def entropy(array):
    sum=0.
    count=len(array)
    vals=set(array)#получаем уникальные значения из массива
    for val in vals:
        temp=float(array.count(val))/count
        if temp==0: continue
        sum-=temp*log(temp,2)
    return round(sum,3)

s0=entropy(balls)
s1=entropy(balls_left)
s2=entropy(balls_right)
#print("balls=%f" % s0) # 9 синих и 11 желтых
print("balls_left=%f" % s1) # 8 синих и 5 желтых
#print("balls_right=%f" % s2) # 1 синий и 6 желтых
print("энтропия игральной кости с несмещенным центром тяжести=%f" % entropy([1,2,3,4,5,6])) # энтропия игральной кости с несмещенным центром тяжести

iqg=round(s0-len(balls_left)/len(balls)*s1-len(balls_right)/len(balls)*s2,3)
print("Каков прирост информации при разделении выборки на balls_left и balls_right - %f" %iqg)
