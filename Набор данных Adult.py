# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 17:52:03 2017
<<<<<<< HEAD


=======
 
#https://github.com/Yorko/mlcourse_open/blob/master/jupyter_notebooks/topic3_decision_trees_knn/hw3_decision_trees.ipynb
>>>>>>> origin/master
"""

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
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
#fig = plt.figure(figsize=(50,30))
#cols = 5
#rows = np.ceil(float(data_train.shape[1]) / cols)
#for i, column in enumerate(data_train.columns):
    #ax = fig.add_subplot(rows, cols, i + 1)
    #ax.set_title(column)
    #if data_train.dtypes[column] == np.object:
    #    data_train[column].value_counts().plot(kind="bar", axes=ax)
    #else:
    #    data_train[column].hist(axes=ax)
    #    plt.xticks(rotation="vertical")
#plt.subplots_adjust(hspace=0.7, wspace=0.2)

#Выяснилось, что в тесте возраст отнесен к типу object, необходимо это исправить.
#print(data_train.dtypes)
#print(data_test.dtypes)


data_test['Age'] = data_test['Age'].astype(int)

#Также приведем показатели типа float в int для соответствия train и test выборок.
data_test['fnlwgt'] = data_test['fnlwgt'].astype(int)
data_test['Education_Num'] = data_test['Education_Num'].astype(int)
data_test['Capital_Gain'] = data_test['Capital_Gain'].astype(int)
data_test['Capital_Loss'] = data_test['Capital_Loss'].astype(int)
data_test['Hours_per_week'] = data_test['Hours_per_week'].astype(int)
<<<<<<< HEAD

#Заполним пропуски в количественных полях медианными значениями, а в категориальных – наиболее часто встречающимся значением
def FillEmptyValues(dataSet):
    categorical_columns = [c for c in dataSet.columns if dataSet[c].dtype.name == 'object']
    numerical_columns = [c for c in dataSet.columns if dataSet[c].dtype.name != 'object']

    for col in categorical_columns:#категориальные заменяем часто встречающейся категорией
        dataSet.loc[dataSet[col]==' ?',col]=dataSet[categorical_columns].describe()[col]['top']

    for col in numerical_columns:#числовые заменяем на медиану
        dataSet[col] = dataSet[col].fillna(dataSet[col].median())

# заполним пропуски
FillEmptyValues(data_train)
FillEmptyValues(data_test)



#Кодируем категориальные признаки 'Workclass', 'Education', 'Martial_Status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Country'.
#Это можно сделать с помощью метода pandas get_dummies.
def ReplaceCatValuesByDummies(dataSet,cols):
    concatCols=[dataSet[dataSet.columns.difference(cols)]]
    for col in cols:
        concatCols.append(pd.get_dummies(dataSet[col], prefix=col))
    return pd.concat(concatCols,axis=1)

cols=['Workclass', 'Education', 'Martial_Status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Country']
data_train=ReplaceCatValuesByDummies(data_train,cols)
data_test=ReplaceCatValuesByDummies(data_test,cols)
del cols

#Проверка расзличности тестовой и обучающей выборки на соотвествие колонок
print(set(data_train.columns) - set(data_test.columns))
#Нет голландии {'Country_ Holand-Netherlands'}
#В тестовой выборке не оказалось Голландии. Заведем необходимый признак из нулей.
data_test['Country_ Holand-Netherlands'] = np.zeros([data_test.shape[0], 1])

#Подготовка закончена, готовым выборки к обучению
X_train=data_train.drop(['Target'], axis=1)
y_train = data_train['Target']

X_test=data_test.drop(['Target'], axis=1)
y_test = data_test['Target']


#Обучите на имеющейся выборке дерево решений (DecisionTreeClassifier) максимальной глубины 3 и получите качество на тесте.
#Используйте параметр random_state = 17 для воспроизводимости результатов.
сlf=DecisionTreeClassifier(criterion='entropy', max_depth=3,random_state = 17)
сlf.fit(X_train,y_train)

#Сделайте с помощью полученной модели прогноз для тестовой выборки.
print(round(accuracy_score(y_test, сlf.predict(X_test)),3))

#Какова доля правильных ответов дерева решений на тестовой выборке при максимальной глубине дерева = 9 и random_state = 17
сlf=DecisionTreeClassifier(criterion='entropy', max_depth=9,random_state = 17)
сlf.fit(X_train,y_train)
=======
>>>>>>> origin/master
