# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 12:59:29 2017


"""
import numpy as np
#import matplotlib.pyplot as plt
import pylab as pl
from scipy import stats
#import seaborn as sns
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler

x = np.linspace(0.0001, 2, 1000)
pl.figure()
_vars=[['g',1,'1'], ['b', 0,'0'], ['m', 1/x, '1/x'] , ['y', 1+x,'1+x']]

for color,koef,leg in _vars:
    y=stats.boxcox(x, lmbda=koef)
    pl.plot(x, y,color,label=leg)

y=np.log(x)
pl.plot(x, y, '.r',label='log')


pl.legend()
pl.xlabel('x')
pl.ylabel('y')
pl.title('График log и boxcox')
pl.show()

print('Задание 1. При каком значении lmbda, выражение np.log(x) == scipy.stats.boxcox(x, lmbda=lmbda) будет истинным. Ответ=0')

print('--------')

with open('train.json', 'r') as raw_data:
    data = json.load(raw_data)

data_train=pd.DataFrame(data=data)
del data

data_train['created']=data_train['created'].apply(pd.to_datetime)
y=data_train['interest_level'].apply(lambda level: 1 if level=='low' else 2 if level=='medium' else 3)
data_train.drop(['interest_level'],axis=1,inplace=True)

#создаем новые признаки на базе поля created
data_train['_created_weekday_24_hour']=data_train['created'].map(lambda created: created.weekday() * 24 + created.hour)
data_train['_created_str_created']=data_train['created'].map(lambda created: str(created))#c последующим one hot encoding
data_train['_created_str_onehot']=pd.factorize(data_train['_created_str_created'])[0]#one-hot кодирование категориального признака
data_train['_created_weekday']=data_train['created'].map(lambda created: created.weekday())
data_train['_created_hour']=data_train['created'].map(lambda created: created.hour)



categorical_columns = [c for c in data_train.columns if data_train[c].dtype.name == 'object']#получаем колонки категориальных характеристик
#categorical_columns = [c for c in data_train.columns if c.startswith('_created')==False]#получаем колонки категориальных характеристик
data_train_float=data_train.drop(categorical_columns,axis=1)#удаляем лишнее
data_train_float.drop(['created'],axis=1,inplace=True)


rf = RandomForestClassifier(n_estimators=50,max_depth =5, n_jobs=-1)#class_weight='balanced'
rf.fit(StandardScaler().fit_transform(data_train_float),y)

featureImportances=pd.DataFrame(data=rf.feature_importances_,index=data_train_float.columns.values)
cols=[c for c in data_train_float.columns if c.startswith('_created')==False]
featureImportances.drop(cols,inplace=True,axis=0)

featureImportances=featureImportances.apply(lambda x: abs(x))
featureImportances.sort_values([0],ascending=True,inplace=True)

print('Задание 2. Какой способ извлечения признаков будет наименее полезным? ',featureImportances.ix[0].name)

#data_train.head()
#data_train.dtypes


print('Задание 3. Какую информацию нельзя извлечь из User-Agent? Разрешение экрана пользователя')
print('Задание 4. Мы решаем задачу классификации: есть пары фотографий, нужно определить, являются ли они фотографиями одного и того же объекта. Какой признак будет наиболее релевантен? Евклидово расстояние между векторами, полученными из предобученной сети ResNet без полносвязных слоев')
print('Задание 5. Для какой из задач отбор признаков (feature selection) будет бесполезен? Борьба со слишком большими значениями признаков, ведущими к переполнению')
print('Задание 6. В каком из районов находится точка с координатами (40.825142, -73.909171)? Bronx')

print('--------------')
print('Задание 7. Обучите класс CountVectorizer из scikit-learn, используя колонку Description из датасета Two Sigma Connect: Rental Listing Inquires таким образом, чтобы в словаре было не более 1000 слов, не меняя прочие параметры по умолчанию. Какое слово соответствует индексу 165 в словаре обученного CountVectorizer?')
cv = CountVectorizer(max_features =1000)
cv.fit_transform(data_train['description'])

for token in ['building','apartment','renthop','trust']:
    if cv.vocabulary_.get(token,0)==165:
        print(token)

del data_train,data_train_float
print('--------------')

from sklearn.base import TransformerMixin
EPSILON = 1e-5

class FeatureEngineer(TransformerMixin):

    def apply(self, df, k, condition):
        df[k] = df['features'].apply(condition)
        df[k] = df[k].astype(np.int8)

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None):
        df = X.copy()

        df.features = df.features.apply(lambda x: ' '.join([y.replace(' ', '_') for y in x]))
        df.features = df.features.apply(lambda x: x.lower())
        df.features = df.features.apply(lambda x: x.replace('-', '_'))

        for k, condition in (('dishwasher', lambda x: 'dishwasher' in x),
                             ('doorman', lambda x: 'doorman' in x or 'concierge' in x),
                             ('pets', lambda x: "pets" in x or "pet" in x or "dog" in x or "cats" in x and "no_pets" not in x),
                             ('air_conditioning', lambda x: 'air_conditioning' in x or 'central' in x),
                             ('parking', lambda x: 'parking' in x),
                             ('balcony', lambda x: 'balcony' in x or 'deck' in x or 'terrace' in x or 'patio' in x),
                             ('bike', lambda x: 'bike' in x),
                             ('storage', lambda x: 'storage' in x),
                             ('outdoor', lambda x: 'outdoor' in x or 'courtyard' in x or 'garden' in x),
                             ('roof', lambda x: 'roof' in x),
                             ('gym', lambda x: 'gym' in x or 'fitness' in x),
                             ('pool', lambda x: 'pool' in x),
                             ('backyard', lambda x: 'backyard' in x),
                             ('laundry', lambda x: 'laundry' in x),
                             ('hardwood_floors', lambda x: 'hardwood_floors' in x),
                             ('new_construction', lambda x: 'new_construction' in x),
                             ('dryer', lambda x: 'dryer' in x),
                             ('elevator', lambda x: 'elevator' in x),
                             ('garage', lambda x: 'garage' in x),
                             ('pre_war', lambda x: 'pre_war' in x or 'prewar' in x),
                             ('post_war', lambda x: 'post_war' in x or 'postwar' in x),
                             ('no_fee', lambda x: 'no_fee' in x),
                             ('low_fee', lambda x: 'reduced_fee' in x or 'low_fee' in x),
                             ('fire', lambda x: 'fireplace' in x),
                             ('private', lambda x: 'private' in x),
                             ('wheelchair', lambda x: 'wheelchair' in x),
                             ('internet', lambda x: 'wifi' in x or 'wi_fi' in x or 'internet' in x),
                             ('yoga', lambda x: 'yoga' in x),
                             ('furnished', lambda x: 'furnished' in x),
                             ('multi_level', lambda x: 'multi_level' in x),
                             ('exclusive', lambda x: 'exclusive' in x),
                             ('high_ceil', lambda x: 'high_ceil' in x),
                             ('green', lambda x: 'green_b' in x),
                             ('stainless', lambda x: 'stainless_' in x),
                             ('simplex', lambda x: 'simplex' in x),
                             ('public', lambda x: 'public' in x),
                             ):
            self.apply(df, k, condition)

        df['bathrooms'] = df['bathrooms'].apply(lambda x: x if x < 5 else 5)
        df['bedrooms'] = df['bedrooms'].apply(lambda x: x if x < 5 else 5)
        df["num_photos"] = df["photos"].apply(len)
        df["num_features"] = df["features"].apply(len)
        created = pd.to_datetime(df.pop("created"))
        df["listing_age"] = (pd.to_datetime('today') - created).apply(lambda x: x.days)
        df["room_dif"] = df["bedrooms"] - df["bathrooms"]
        df["room_sum"] = df["bedrooms"] + df["bathrooms"]
        df["price_per_room"] = df["price"] / df["room_sum"].apply(lambda x: max(x, .5))
        df["bedrooms_share"] = df["bedrooms"] / df["room_sum"].apply(lambda x: max(x, .5))
        df['price'] = df['price'].apply(lambda x: np.log(x + EPSILON))

        key_types = df.dtypes.to_dict()
        for k in key_types:
            if key_types[k].name not in ('int64', 'float64', 'int8'):
                df.pop(k)

        for k in ('latitude', 'longitude', 'listing_id'):
            df.pop(k)
        return df


def encode(x):
    if x == 'low':
        return 0
    elif x == 'medium':
        return 1
    elif x == 'high':
        return 2


def get_data():
    with open('train.json', 'r') as raw_data:
        data = json.load(raw_data)

    df = pd.DataFrame(data)
    target = df.pop('interest_level').apply(encode)

    df = FeatureEngineer().fit_transform(df)
    return df, target

x_train, y=get_data()
mms=MinMaxScaler()
x_train_mms=pd.DataFrame(data=mms.fit_transform(x_train))#шкалируем
_vars=pd.DataFrame(data=x_train_mms.var())
good_select=_vars>=.1
print('Задание 8. Сколько признаков осталось в датасете?',good_select[good_select[0]].count()[0])



