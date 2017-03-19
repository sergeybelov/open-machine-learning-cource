# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 17:52:03 2017
 
#https://github.com/Yorko/mlcourse_open/blob/master/jupyter_notebooks/topic3_decision_trees_knn/hw3_decision_trees.ipynb
"""

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
#==============================================================================
# Набор данных UCI Adult (качать не надо, все есть в репозитории): классификация людей с
# помощью демографических данных для прогнозирования, зарабатывает ли человек более $ 50 000 в год.
#
# Описание признаков:
# Age – возраст, количественный признак
# Workclass – тип работодателя, количественный признак
# fnlwgt – итоговый вес обьекта, количественный признак
# Education – уровень образования, качественный признак
# Education_Num – количество лет обучения, количественный признак
# Martial_Status – семейное положение, категориальный признак
# Occupation – профессия, категориальный признак
# Relationship – тип семейных отношений, категориальный признак
# Race – раса, категориальный признак
# Sex – пол, качественный признак
# Capital_Gain – прирост капитала, количественный признак
# Capital_Loss – потери капитала, количественный признак
# Hours_per_week – количество часов работы в неделю, количественный признак
# Country – страна, категориальный признак
# Целевая переменная: Target – уровень заработка, категориальный (бинарный) признак
#==============================================================================
data_train = pd.read_csv('adult.data.csv',header=None)
data_train.columns=['Age','Workclass','fnlwgt','Education','Education_Num','Martial_Status','Occupation','Relationship','Race','Sex','Capital_Gain','Capital_Loss','Hours_per_week','Country','Target']
data_test = pd.read_csv('adult.test.csv',header=None)
data_test.columns=['Age','Workclass','fnlwgt','Education','Education_Num','Martial_Status','Occupation','Relationship','Race','Sex','Capital_Gain','Capital_Loss','Hours_per_week','Country','Target']

# необходимо убрать строки с неправильными метками в тестовой выборке
data_test = data_test[(data_test['Target'] == ' >50K.') | (data_test['Target']==' <=50K.')]

# перекодируем target в числовое поле
data_train.Target=data_train.Target.map(lambda x: 0 if x.lstrip().startswith('<=50K') else 1)
data_test.Target=data_test.Target.map(lambda x: 0 if x.lstrip().startswith('<=50K') else 1)


#Рисуем график
fig = plt.figure(figsize=(50,30))
cols = 5
rows = np.ceil(float(data_train.shape[1]) / cols)
for i, column in enumerate(data_train.columns):
    ax = fig.add_subplot(rows, cols, i + 1)
    ax.set_title(column)
    if data_train.dtypes[column] == np.object:
        data_train[column].value_counts().plot(kind="bar", axes=ax)
    else:
        data_train[column].hist(axes=ax)
        plt.xticks(rotation="vertical")
plt.subplots_adjust(hspace=0.7, wspace=0.2)

#Выяснилось, что в тесте возраст отнесен к типу object, необходимо это исправить.
print(data_train.dtypes)
print(data_test.dtypes)


data_test['Age'] = data_test['Age'].astype(int)

#Также приведем показатели типа float в int для соответствия train и test выборок.
data_test['fnlwgt'] = data_test['fnlwgt'].astype(int)
data_test['Education_Num'] = data_test['Education_Num'].astype(int)
data_test['Capital_Gain'] = data_test['Capital_Gain'].astype(int)
data_test['Capital_Loss'] = data_test['Capital_Loss'].astype(int)
data_test['Hours_per_week'] = data_test['Hours_per_week'].astype(int)
