# -*- coding: utf-8 -*-
"""
Редактор Spyder
"""

#импортируем необходимые библиотеки и скачаем данные
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer

#В зависимости от версии sklearn
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

import numpy as np
import scipy


from matplotlib import pyplot as plt

train_df = pd.read_csv('howpop_train.csv')
test_df  = pd.read_csv('howpop_test.csv')

#Убедимся, что данные отсортированы по признаку published
#train_df['published'].apply(lambda ts: pd.to_datetime(ts).value).plot();

train_df.published = train_df.published.apply(pd.to_datetime)#Кодируем в формат datetime64 заданные колонки
corr_matrix=train_df.corr()

#Есть ли в train_df признаки, корреляция между которыми больше 0.9? - НЕТ
#Обратите внимание, именно различные признаки - корреляция признака с самим собой естественно больше 0.9 :)

#==============================================================================
# В каком году было больше всего публикаций? (Рассматриваем train_df)
# 2014
# 2015 <-
# 2016
# 2017
#==============================================================================
train_df['year']=train_df.published.map(lambda x: x.year)#получаем данные по году
yearpublished=pd.DataFrame(data=train_df.groupby('year')['url'].count())
yearpublished.sort_values('url',ascending=False,inplace=True)

print(u'В каком году было больше всего публикаций? (Рассматриваем train_df)', yearpublished.iloc[0])
#Name: 2015, dtype: int64


#Разбиение на train/valid
#Используем только признаки 'author', 'flow', 'domain' и 'title'
features = ['author', 'flow', 'domain','title']
train_size = int(0.7 * train_df.shape[0])#Размер обучающей выборки

X, y = train_df.ix[:, features],  train_df['favs_lognorm'] #отделяем признаки от целевой переменной
X_test = test_df.ix[:, features]

X_train, X_valid = X.iloc[:train_size, :], X.iloc[train_size:,:]
y_train, y_valid = y.iloc[:train_size], y.iloc[train_size:]


#==============================================================================
# Основные параметры TfidfVectorizer в sklearn:
# min_df - при построении словаря слова, которые встречаются реже, чем указанное значение, игнорируются
# max_df - при построении словаря слова, которые встречаются чаще, чем указанное значение, игнорируются
# analyzer - определяет, строятся ли признаки по словам или по символам (буквам)
# ngram_range - определяет, формируются ли признаки только из отдельных слов или из нескольких слов
# (в случае с analyzer='char' задает количество символов). Например, если указать analyzer='word' и
# ngram_range=(1,3),то признаки будут формироваться из отдельных слов, из пар слов и из троек слов.
# stop_words - слова, которые игнорируются при построении матрицы
# Более подробно с параметрами можно ознакомиться в документации
# Инициализируйте TfidfVectorizer с параметрами min_df=3, max_df=0.3 и ngram_range=(1, 3).
# Примените метод fit_transform к X_train['title'] и метод transform к X_valid['title'] и X_test['title']
#==============================================================================
def getTfid(analyser):
    tfid=TfidfVectorizer(min_df=3, max_df=0.3, ngram_range=(1, 3))
    if analyser: tfid=TfidfVectorizer(analyzer='char')
    tfid.fit_transform(X_train['title'])
    tfid.transform(X_valid['title'])
    tfid.transform(X_test['title'])
    return tfid

tfid=getTfid(False)
vocabulary=tfid.vocabulary_
print(u'Размер словаря: ',len(vocabulary))
#Вопрос 4.2.3. Какой размер у полученного словаря?
#43789
#50624 <-
#93895
#74378



#Какой индекс у слова 'python'?
#1
#10
#9065 <-
#15679
print(u'Какой индекс у слова "python"?',vocabulary['python'])


#==============================================================================
# Инициализируйте TfidfVectorizer, указав analyzer='char'.
# Примените метод fit_transform к X_train['title'] и метод transform к X_valid['title'] и X_test['title']
# Какой размер у полученного словаря?
# 218 <-
# 510
# 125
# 981
#==============================================================================
tfid=getTfid(True)
print(u'Размер словаря: ',len(tfid.vocabulary_))
#Размер словаря:  218

#Работа с категориальными признаками
#Для обработки категориальных признаков 'author', 'flow', 'domain'
#мы будем использовать DictVectorizer из sklearn.
feats = ['author', 'flow', 'domain']


vectorizer_title=TfidfVectorizer(min_df=3, max_df=0.3, ngram_range=(1, 3))
X_train_title=vectorizer_title.fit_transform(X_train['title'])
X_valid_title=vectorizer_title.transform(X_valid['title'])
X_test_title=vectorizer_title.transform(X_test['title'])


vectorizer_title_ch=TfidfVectorizer(analyzer='char')
X_train_title_ch=vectorizer_title.fit_transform(X_train['title'])
X_valid_title_ch =vectorizer_title.transform(X_valid['title'])
X_test_title_ch=vectorizer_title.transform(X_test['title'])
#==============================================================================
# Инициализируйте DictVectorizer с параметрами по умолчанию.
# Примените метод fit_transform к признакам 'author', 'flow', 'domain'
# X_train и метод transform к тем же признакам X_valid и X_test
#==============================================================================
#1. сначала заполняем пропуски прочерком
#2. реобразуем датафрейм в словарь, где ключами являются индексы объектов (именно для этого мы транспонировали датафрейм),
#а значениями являются словари в виде 'название_колонки':'значение'
#3. В DictVectorizer нам нужно будет передать список словарей для каждого объекта
#в виде 'название_колонки':'значение',
#поэтому используем .values()
vectorizer_feats =DictVectorizer()
X_train_feats = vectorizer_feats.fit_transform(X_train[feats].fillna('-').T.to_dict().values())
X_valid_feats = vectorizer_feats.transform(X_valid[feats].fillna('-').T.to_dict().values())
X_test_feats = vectorizer_feats.transform(X_test[feats].fillna('-').T.to_dict().values())

#Соединим все полученные матрицы при помощи scipy.sparse.hstack()
X_train_new = scipy.sparse.hstack([X_train_title, X_train_feats, X_train_title_ch])
X_valid_new = scipy.sparse.hstack([X_valid_title, X_valid_feats, X_valid_title_ch])
X_test_new =  scipy.sparse.hstack([X_test_title, X_test_feats, X_test_title_ch])


#==============================================================================
# Обучите две модели на X_train_new, y_train, задав в первой alpha=0.1 и random_state = 1,
# а во второй alpha=1.0 и random_state = 1
# Рассчитайте среднеквадратичную ошибку каждой модели (mean_squared_error).
#Сравните значения ошибки на обучающей и тестовой выборках и ответьте на вопросы.
# Выберите верные утверждения:
# обе модели показывают одинаковый результат (среднеквадратичная ошибка отличается не больше чем на тысячные),
#    регуляризация ничего не меняет
# при alpha=0.1 модель переобучается <-
# среднеквадратичная ошибка первой модели на тесте меньше
# при alpha=1.0 у модели обощающая способность лучше, чем у при alpha=0.1 <-
#==============================================================================
model1=Ridge(alpha=0.1, random_state = 1)
model1.fit(X_train_new,y_train)

train_preds1 = model1.predict(X_train_new)
valid_preds1 = model1.predict(X_valid_new)

print('alpha=0.1. Ошибка на трейне',mean_squared_error(y_train, train_preds1))
print('alpha=0.1. Ошибка на тесте',mean_squared_error(y_valid, valid_preds1))

model2=Ridge(alpha=1.0, random_state = 1)
model2.fit(X_train_new,y_train)
train_preds2 = model2.predict(X_train_new)
valid_preds2 = model2.predict(X_valid_new)

print('alpha=1.0. Ошибка на трейне',mean_squared_error(y_train, train_preds2))
print('alpha=1.0. Ошибка на тесте',mean_squared_error(y_valid, valid_preds2))