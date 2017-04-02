# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 12:07:02 2017

@author: Iru
"""
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

## Сделаем функцию, которая будет заменять NaN значения на медиану в каждом столбце таблицы
def delete_nan(table):
    for col in table.columns:
        table[col].fillna(table[col].median(), inplace=True)
    return table

#==============================================================================
#Вопрос 5.1.  В зале суда есть 5 присяжных, каждый из них по отдельности с вероятностью 70% может правильно определить,
# виновен подсудимый или нет. С какой вероятностью они все вместе вынесут правильный вердикт,
# если решение принимается большинством голосов?
#==============================================================================
def calc(p,i,N):
    return (1-p)**(N-i)

def soch(i,N):
    return math.factorial(N) / (math.factorial(i)*math.factorial(N-i))

def thisSum(p,i,N):
    return soch(i,N)* p**i *calc(p,i,N)

sum=0
N=5
m=int((N+1)/2)
p=0.7
for i in range(m,N+1):
    sum+=thisSum(p,i,N)

print(u'Вопрос 5.1.:',round(sum*100,1))


#==============================================================================
# Вопрос 5.2. Какова интервальную оценка среднего возраста
# (age) для клиентов, которые просрочили выплату кредита, с 90% "уверенностью"?
#==============================================================================
#Прогнозируемая переменная
#SeriousDlqin2yrs ----Человек не выплатил данный кредит в течение 90 дней; возможные значения 1/0 (не выплатил/выплатил)
#Независимые признаки
#age ---- Возраст заёмщика кредитных средств; тип - integer
#NumberOfTime30-59DaysPastDueNotWorse ----Количество раз, когда человек имел
# просрочку выплаты других кредитов более 30-59 дней, но не больше в течение последних двух лет; тип - integer
#DebtRatio ---- Ежемесячный отчисления на задолжености(кредиты,алименты и т.д.) / совокупный месячный доход percentage; тип - real
#MonthlyIncome ----Месячный доход в долларах; тип - real
#NumberOfTimes90DaysLate ----Количество раз, когда человек имел просрочку выплаты других кредитов более 90 дней; тип - integer
#NumberOfTime60-89DaysPastDueNotWorse---- Количество раз, когда человек имел
#просрочку выплаты других кредитов более 60-89 дней, но не больше в течение последних двух лет; ; тип - integer
#NumberOfDependents ----Число человек в семье кредитозаёмщика; тип - integer

## Считываем данные
data = pd.read_csv('credit_scoring_sample.csv', sep =';')

## Рассмотрим типы считанных данных
#data.dtypes

## Посмотрим на распределение классов в зависимой переменной
#НУЖНО ДЛЯ ОПРЕДЕЛЕНИЕ БАЛАНСА В КЛАССАХ!!
#В ПРИМЕРЕ КЛАССЫ РАЗБАЛАНСИРОВАНЫ
ax =data['SeriousDlqin2yrs'].hist(orientation='horizontal', color='red')
ax.set_xlabel("number_of_observations")
ax.set_ylabel("unique_value")
ax.set_title("Target distribution")

print('Distribution of target')
print(data['SeriousDlqin2yrs'].value_counts()/data.shape[0])
print('-- В ПРИМЕРЕ КЛАССЫ РАЗБАЛАНСИРОВАНЫ --')

## Выберем названия всех признаков из таблицы, кроме прогнозируемого
independent_columns_names = data.columns.difference(['SeriousDlqin2yrs']).tolist()

## Применяем функцию, заменяющую все NaN значения на медианное значение соответствующего столбца
table=delete_nan(data)

## Разделяем таргет и признаки
X =table[independent_columns_names]
y =table['SeriousDlqin2yrs']

#==============================================================================
# Задание 2. Сделайте интервальную оценку среднего возраста (age) для клиентов, которые просрочили выплату кредита, с 90%
# "уверенностью". (используйте пример из статьи. Поставьте np.random.seed(0) как это сделано в статье)
#==============================================================================
def get_bootstrap_samples(data, n_samples):
    # функция для генерации подвыборок с помощью бутстрэпа
    indices = np.random.randint(0, len(data), (n_samples, len(data)))
    samples = data[indices]
    return samples

def stat_intervals(stat, alpha):
    # функция для интервальной оценки
    boundaries = np.percentile(stat, [100 * alpha / 2., 100 * (1 - alpha / 2.)])
    return np.around(boundaries, 2)

# ставим seed для воспроизводимости результатов
np.random.seed(0)

#плохие заемщики
badDebedsAge=table[table['SeriousDlqin2yrs'] == 1]['age'].values
badDebeds_mean_scores = [np.mean(sample)
                       for sample in get_bootstrap_samples(badDebedsAge, 1000)]
print("Вопрос 5.2",  stat_intervals(badDebeds_mean_scores, 0.10))

#==============================================================================
# Вопрос 5.3. Какое оптимальное значение параметра С?
#==============================================================================
#==============================================================================
# Подбор параметров для модели логистической регрессии
# Одной из важных метрик качества модели является значение площади под ROC-кривой.
# Значение ROC-AUC лежит от 0 до 1. Чем ближе начение метрики ROC-AUC к 1, тем качественнее происходит классификация моделью.
#==============================================================================

## Используем модуль LogisticRegression для построения логистической регрессии.
## Из-за несбалансированности классов  в таргете добавляем параметр балансировки.
## Используем также параметр random_state=5 для воспроизводимости результатов
lr = LogisticRegression(random_state=5, class_weight= 'balanced')

## Попробуем подобрать лучший коэффициент регуляризации (коэффициент C в логистической регрессии) для модели лог.регрессии.
## Этот параметр необходим для того, чтобы подобрать оптимальную модель, которая не будет переобучена, с одной стороны,
## и будет хорошо предсказывать значения таргета, с другой.
## Остальные параметры оставляем по умолчанию.
parameters = {'C': (0.0001, 0.001, 0.01, 0.1, 1, 10)}

## Для того, чтобы подобрать коэффициент регуляризации, попробуем для каждого его возможного значения посмотреть
## значения roc-auc на стрэтифайд кросс-валидации из 5 фолдов с помощью функции StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)

#==============================================================================
# Задание 3. Сделайте GridSearch с метрикой "roc-auc" по параметру C.
# Какое оптимальное значение параметра С?
#==============================================================================
clf_grid = GridSearchCV(lr, parameters,cv=skf, n_jobs=1,verbose=1,scoring='roc_auc')
clf_grid.fit(X,y)
print('Вопрос 5.3. Какое оптимальное значение параметра С? ',clf_grid.best_params_['C'])
#print("best_score")
#print(clf_grid.best_score_)

def calc_score(md, text):
    scores = cross_val_score(md, X,y, scoring='roc_auc', cv=skf)#Оценка алгоритма
    val=round(scores.mean()*100,2)#берем среднее значение оценки
    print("Оценка качества (",text,") ",val)

lr = LogisticRegression(random_state=5, class_weight= 'balanced',**clf_grid.best_params_)
calc_score(lr,'LogisticRegression')

#==============================================================================
# Задание 4. Можно ли считать лучшую модель устойчивой? (модель считаем устойчивой,
# если стандартное отклонение на валидации меньше 0.5%)
# Сохраните точность лучшей модели, она вам приходится для следующих заданий
#==============================================================================
bestIndex=clf_grid.best_index_#Индекс лучшей модели

print('Вопрос 5.4. Можно ли считать лучшую модель устойчивой? ' 'Да' if clf_grid.cv_results_['std_test_score'][bestIndex]*100<0.5 else 'Нет')

#==============================================================================
# Задание 5. Определите самый важный признак.
# Важность признака определяется абсолютным значением его коэффициента.
# Так же нужно нормализировать все признаки, что бы можно их было корректно сравнить.
#==============================================================================
X_norm=StandardScaler().fit_transform(X)

lr=LogisticRegression(random_state=5, class_weight= 'balanced', **clf_grid.best_params_)
lr.fit(X_norm, y)#Обучаем

#получаем список показателей которые сильнее всего влияют на предсказания
featureImportances=pd.DataFrame(data=lr.coef_,columns=independent_columns_names).T
featureImportances=featureImportances.apply(lambda x: abs(x))
featureImportances.sort_values([0],ascending=False,inplace=True)
print('Задание 5. Определите самый важный признак. Важность признака определяется абсолютным значением его коэффициента. ',featureImportances.ix[0].name)

#==============================================================================
# Задание 6. Посчитайте долю влияния DebtRatio на предсказание. (Воспользуйтесь функцией softmax)
#==============================================================================
def softmax(x):
    max_ = np.max(x)
    e_x = np.exp(x - max_)
    return e_x / e_x.sum()

vals=pd.DataFrame(data=softmax(lr.coef_),columns=independent_columns_names).T
vals=vals.apply(lambda x: round(x,2))
print('Задание 6. Посчитайте долю влияния DebtRatio на предсказание.',vals.loc['DebtRatio',0])

#==============================================================================
# Задание 7. Давайте посмотрим как можно интерпретировать влияние наших признаков.
# Для этого заного оценим логистическую регрессию в абсолютных величинах.
# После этого посчитайте во сколько раз увеличатся шансы, что клиент не выплатит кредит,
# если увеличить возраст на 20 лет при всех остальных равных значениях признаков.
#==============================================================================
lr=LogisticRegression(random_state=5, class_weight= 'balanced', **clf_grid.best_params_)
lr.fit(X, y)#Обучаем

featureImportances=pd.DataFrame(data=lr.coef_,columns=independent_columns_names).T
_or=round(math.exp(20*featureImportances.loc['age'][0]),2)#умножаем коэффициент на 20 лет

print('Вопрос 5.7. Во сколько раз увеличатся шансы, что клиент не выплатит кредит, если увеличить возраст на 20 лет при всех остальных неизменных значениях признаков. ',_or)

print('----------------------')

#==============================================================================
# Задание 8. На сколько точность лучшей модели случайного леса выше точности логистической регрессии на валидации?
#==============================================================================
# Инициализируем случайный лес с 100 деревьями и сбалансированными классами
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42, oob_score=True, class_weight='balanced')

## Будем искать лучшие параметры среди следующего набора
parameters = {'max_features': [1, 2, 4], 'min_samples_leaf': [3, 5, 7, 9], 'max_depth': [5,10,15]}

## Делаем опять же стрэтифайд k-fold валидацию. Инициализация которой должна у вас продолжать храниться в skf
clf_grid = GridSearchCV(rf, parameters,cv=skf, n_jobs=1,verbose=1,scoring='roc_auc')
clf_grid.fit(X,y)

rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42, oob_score=True, class_weight='balanced',**clf_grid.best_params_)
calc_score(rf,'RandomForestClassifier')
