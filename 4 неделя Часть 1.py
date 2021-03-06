# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 13:52:12 2017

"""
#==============================================================================
# Вопрос 1, не опечатка ли в формулировке?
# - опечатки нет, выведите размерности датафреймов в том порядке, который в вопросе.
# Вопрос 2, в чем подвох с ниболее посещаемыми сайтами Элис?
# - подвоха нет, посмотрите пример выше и сделайте аналогично. Разнесите сайты по категориям и посчитайте суммарное количество посещений.
# Вопрос 3, что надо рассчитать, чтобы ответить? Как сравнивать величины?
# - на первой неделе вы научились как описать числовой признак различными статистиками, больше ничего не потребуется.
# Чтобы сравнить величины, посмотрите, как они изменяются друг относительно друга. Пусть у нас есть признак, который принимает значения sin(x), минимальное значение признака -1, а максимальное +1, сам признак изменяется в диапазоне от -1 до 1. Пусть есть второй признак, 2*cos(x), он, очевидно, изменяется в диапазоне от -2 до 2. Хотя для нас эти значения небольшие, диапазон первого признака в два раза уже, чем у второго. Максимальное значение первого в два раза меньше максимального значения второго. Когда величиных отличаются в несколько раз их нельзя назвать (примерно) одинаковыми.
# Вопрос 4, надо ли убирать столбец с нулевыми сайтами?
# - перечитайте последнее предложение перед вопросом.
# Вопрос 5, есть два очень близких варианта, так и должно быть?
# - так и должно, напишите строчку кода и ответ сам посмотрит на вас.
# Вопрос 6, у меня совсем странный график
# - возможно, вы используете индексы датафрейма, либо откажитесь от них, либо работайте только с обучающей выборкой.
# Вопрос 7, если качество уменьшилось, не все ли равно масштабируем признак или нет?
# - выберите вариант, где качество лучше. Не забудьте, что надо использовать признак start_month, который в таблице full_new_feat хранится в сыром виде (а в каком случае от него наибольшая польза?)
# Вопрос 7, надо ли считать нулевые сайты?
# - можно 0 тоже считать уникальным сайтом. Хотя это не совсем правильно, результат качественно отличаться не будет. Если не хотите считать нулевые сайты, воспользуйтесь методом nonzero() из пандас.
# Вопрос 8, какие варианты надо перебрать?
# - во-первых, используйте подсказки из предыдущего вопроса. Во-вторых, переберите все возможные комбинации двух новых признаков (сырой/масштабированный), их не так много.
# Вопрос 9, мой оптимальный коэффициент C отличается от предложенных вариантов
# - округлите свой ответ до двух знаков, если и теперь отличается, напишите в личку.
# Был ли интернет во времена Христа?
# - есть все основания полагать, что не было.
#==============================================================================


# загрузим библиотеки и установим опции
from __future__ import division, print_function
from matplotlib import pyplot as plt
import seaborn as sns


import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression


#загрузим обучающую и тестовую выборки
#признаки site_i – это индексы посещенных сайтов (расшифровка дана в pickle-файле со словарем site_dic.pkl).
#Признаки time_j – время посещения сайтов site_j.
#Целевой признак target – факт того, что сессия принадлжит Элис (то есть что именно Элис ходила по всем этим сайтам).
train_df = pd.read_csv('train_sessions.csv',
                       index_col='session_id')
test_df = pd.read_csv('test_sessions.csv',
                      index_col='session_id')

#Чтобы понять с какими данными имеем дело используем info
#train_df.info()

# приведем колонки time1, ..., time10 к временному формату
times = ['time%s' % i for i in range(1, 11)]#формируме массив колонок содержащих время
train_df[times] = train_df[times].apply(pd.to_datetime)#Кодируем в формат datetime64 заданные колонки
test_df[times] = test_df[times].apply(pd.to_datetime)#Кодируем в формат datetime64 заданные колонки

# отсортируем данные по времени
train_df = train_df.sort_values(by='time1')

#==============================================================================
# В обучающей выборке содержатся следующие признаки:
# - site1 – индекс первого посещенного сайта в сессии
# - time1 – время посещения первого сайта в сессии
# - ...
# - site10 – индекс 10-го посещенного сайта в сессии
# - time10 – время посещения 10-го сайта в сессии
# - target – целевая переменная, принимает значение 1 для сессий Элис-хакер и 0 для сессий других пользователей

# Сессии пользователей выделены таким образом, что они не могут быть длинее получаса или содержит более 10 сайтов.
# То есть сессия считается оконченной либо когда пользователь посетил 10 сайтов подряд, либо когда сессия заняла
# по времени более 30 минут.
# В таблице встречаются пропущенные значения, это значит, что сессия состоит менее, чем из 10 сайтов.
#==============================================================================

# Заменим пропущенные значения нулем и приведем колонки целому типу (сейчас float).
# приведем колонки site1, ..., site10 к целочисленному формату и заменим пропуски нулями
sites = ['site%s' % i for i in range(1, 11)]
train_df[sites] = train_df[sites].fillna(0).astype('int')
test_df[sites] = test_df[sites].fillna(0).astype('int')

# загрузим словарик сайтов
with open(r"site_dic.pkl", "rb") as input_file:
    site_dict = pickle.load(input_file)

#Также загрузим словарь сайтов и посмотрим как он выглядит:
# датафрейм словарика сайтов
sites_dict = pd.DataFrame(list(site_dict.keys()), index=list(site_dict.values()), columns=['site'])
print(u'всего сайтов:', sites_dict.shape[0])#показывает размер датасета


#==============================================================================
# Задание 1: Какие размерности
# имеют тестовая и обучающая выборки?
# (82797, 20) (253561, 20)
# (82797, 20) (253561, 21)  <- с включением целевой  переменной
# (253561, 21) (82797, 20)
# (253561, 20) (82797, 20)
#==============================================================================

print(u'test_df= ',test_df.shape)#len(test_df),'x',len(test_df.columns.values))
print(u'train_df= ',train_df.shape)#len(train_df),'x',len(train_df.columns.values))



#==============================================================================
# Задание 2: Какие сайты Элис посещает в сети наиболее часто?
# видеохостинги <-
# социальные сети
# торрент-трекеры
# новостные сайты
#==============================================================================
ellisSites=train_df[train_df['target']==1]
top_sites = pd.Series(ellisSites[sites].fillna(0).values.flatten()
                     ).value_counts().sort_values(ascending=False).head(3)
print('Какие сайты Элис посещает в сети наиболее часто?')
print(sites_dict.ix[top_sites.index])

#==============================================================================
# 77      i1.ytimg.com
# 80     s.youtube.com
# 76   www.youtube.com
#==============================================================================

# создадим отдельный датафрейм, где будем работать со временем
time_df = pd.DataFrame(index=train_df.index)
time_df['target'] = train_df['target']

# найдем время начала и окончания сессии
time_df['min'] = train_df[times].min(axis=1)
time_df['max'] = train_df[times].max(axis=1)

# вычислим длительность сессии и переведем в секунды
time_df['seconds'] = (time_df['max'] - time_df['min']) / np.timedelta64(1, 's')

#==============================================================================
# Задание 3. Выберите все верные утверждения (может оказаться один верный ответ, несколько или ни одного):
# в среднем сессия Элис короче, чем у остальных пользователей - Да: 52/139 секунд
# доля сессий Элис в выборке больше 1% - Нет.
# диапазоны длительности сессий и Элис, и остальных примерно одинаковы - Да: Пользователь [0.0-1800.0], Эллис [0.0-1763.0]
# разброс значений относительно среднего у всех пользователей (Элис в том числе) приблизительно одинаков -  Нет
# доля сессий Элис от 40 секунд и дольше составляет менее четверти - Нет
#==============================================================================
print('D среднем сессия Элис короче, чем у остальных пользователей')
print(u'Ellis session lasts:', time_df[time_df.target==1].seconds.mean())
print(u'General user session lasts:', time_df[time_df.target==0].seconds.mean())

onePercent=time_df.target.count()/100
EllisSessions=len(ellisSites.target)
print(u'Доля сессий Элис в выборке больше 1%?', 'Да' if EllisSessions>onePercent else 'Нет')


def GetRange(target):
    select=time_df[time_df.target==target]
    minVal = select.seconds.min()
    maxVal = select.seconds.max()
    print('Target: %s [%s-%s]' %(target,minVal,maxVal))

print('Диапазоны длительности сессий и Элис, и остальных примерно одинаковы')
GetRange(0)
GetRange(1)


print('разброс значений относительно среднего у всех пользователей (Элис в том числе) приблизительно одинаков')
sns.boxplot(x="seconds", data=time_df)
#meanSeconds=time_df.seconds.mean()
#pic=time_df.seconds.plot(kind='bar',title='Значения сессий')
#pic.plot(0, meanSeconds, label = u'Среднее')
#time_df['mean_deviance']=time_df.seconds-meanSeconds
#time_df['mean_deviance']=time_df['mean_deviance'].map(lambda x: math.fabs(x))

#Реиндексация
#time_df.index=range(1,len(time_df)+1)
#sns.distplot(time_df.seconds,kde=False)

print('доля сессий Элис от 40 секунд и дольше составляет менее четверти. ')
session40Sec=len(time_df[(time_df.seconds>=40) & (time_df.target==1)])
print ('Да') if session40Sec<onePercent*25 else print ('Нет')


# наша целевая переменная
y_train = train_df['target']

# объединенная таблица исходных данных
full_df = pd.concat([train_df.drop('target', axis=1), test_df])

# индекс, по которому будем отделять обучающую выборку от тестовой
idx_split = train_df.shape[0]#индекс последнего элемента


#==============================================================================
# Для самой первой модели будем использовать только посещенные сайты в сессии
# (но не будем обращать внимание на временные признаки).
# За таким выбором данных для модели стоит следующая идея: у Элис есть свои излюбленные сайты,
# и чем чаще вы видим эти сайты в сессии, тем выше вероятность, что это сессия Элис, и наоборот.
# Подготовим данные, из всей таблицы выберем только признаки site1, site2, ... , site10.
# Напомним, что пропущенные значения заменены нулем. Вот как выглядят первые строки таблицы:
#==============================================================================
# табличка с индексами посещенных сайтов в сессии
full_sites = full_df[sites]

# последовательность с индексами
sites_flatten = full_sites.values.flatten()#Преобразовываем матрицу в массив

# искомая матрица
full_sites_sparse = csr_matrix(([1] * sites_flatten.shape[0],#Какими значениями заполнять ненулевые данные разреженной матрицы
                                sites_flatten,#Индексы значений в разреженной матрице для заполнения
                                range(0, sites_flatten.shape[0]  + 10, 10)))[:, 1:]# индексы разбиения на строки для построения разреженной матрици, например, строка 0 это элементы между индексами [0; 3) - крайнее правое значение не включается  строка 1 это элементы между индексами [3; 6)
del sites_flatten
print('Задание 4: А теперь еще один вопрос, чему равна разреженность матрицы из минипримера?')
#==============================================================================
# отношение количества нулевых элементов к общему числу элементов называется разреженностью матрицы.
# 42% <-
# 47%
# 50%
# 53%
#==============================================================================
elemsCount=full_sites_sparse.nnz
print(round((elemsCount-full_sites_sparse.count_nonzero())/elemsCount*100,0))

#==============================================================================
# 3. Построение первой модели
# Итак, у нас есть алгоритм и данные для него, построим нашу первую модель, воспользовавшись реализацией логистической регрессии
# из пакета sklearn с параметрами по умолчанию.
# Первые 90% данных будем использовать для обучения (обучающая выборка отсортирована по времени), а
# оставшиеся 10% для проверки качества (validation). Напишем
# простую функцию, которая будет возвращать качество модели и обучим наш первый классификатор
#
#==============================================================================
def get_auc_lr_valid(X, y, C=1.0, seed=17, ratio = 0.9):
    # разделим выборку на обучающую и валидационную
    idx = int(round(X.shape[0] * ratio))
    # обучение классификатора
    lr = LogisticRegression(C=C, random_state=seed, n_jobs=-1).fit(X[:idx, :], y[:idx])
    # прогноз для валидационной выборки
    y_pred = lr.predict_proba(X[idx:, :])[:, 1]
    # считаем качество
    score = roc_auc_score(y[idx:], y_pred)

    return score

# выделим из объединенной выборки только обучающую (для которой есть ответы)
X_train = full_sites_sparse[:idx_split, :]

print('Точность обученной модели')#0.919524105836 - первый бейслайн
# считаем метрику на валидационной выборке
print(get_auc_lr_valid(X_train, y_train))

# функция для записи прогнозов в файл
def write_to_submission_file(predicted_labels, out_file,
                             target='target', index_label="session_id"):
    predicted_df = pd.DataFrame(predicted_labels,
                                index = np.arange(1, predicted_labels.shape[0] + 1),
                                columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)


# обучим модель на всей выборке
# random_state=17 для воспроизводимости
# параметр C=1 по умолчанию, но здесь мы его укажем явно
lr = LogisticRegression(C=1.0, random_state=17).fit(X_train, y_train)

# сделаем прогноз для тестовой выборки
X_test = full_sites_sparse[idx_split:,:]
y_test = lr.predict_proba(X_test)[:, 1]

# запишем его в файл, готовый для сабмита
write_to_submission_file(y_test, 'baseline_1.csv')

#==============================================================================
# Задание 5: данные за какие годы представлены в обучающей и тестовой выборке?
# за 13 и 14
# за 2012 и 2013
# за 2013 и 2014 <-
# за 2014 и 2015
#==============================================================================
#Делаем конкатенацию всех временных колонок в одну колонку
cols=[]
for col in times:
    cols.append(full_df[col])
times_df = pd.DataFrame(data=pd.concat(cols,axis=0))
times_df['year']=times_df[0].map(lambda x: x.year)#получаем данные по году
print('данные за какие годы представлены в обучающей и тестовой выборке?')
years=times_df.groupby('year')['year'].count()
print(years)
del times_df

#==============================================================================
# Создадим такой признак, который будет представлять из себя число вида ГГГГММ от той даты, когда проходила сессия,
# например 201407 -- 2014 год и 7 месяц. Таким образом мы будем учитывать помесячный линейный тренд за весь период предоставленных данных.
#==============================================================================
# датафрейм для новых признаков
full_new_feat = pd.DataFrame(index=full_df.index)

# добавим признак start_month
full_new_feat['start_month'] = full_df['time1'].apply(lambda ts: 100 * ts.year + ts.month)#тип int
#==============================================================================
# Задание 6: Постройте график количества сессий Элис в зависимости от новой переменной start_month. Выберите верное утверждение:
# Элис вообще не выходила в сеть за все это время -  Нет
# с начала 2013 года по середину 2014 года количество ежемесячных сессий уменьшилось - Нет
# в целом количество сессий Элис за месяц постоянно на протяжении всего периода - Нет
# с начала 2013 года по середину 2014 года количество ежемесячных сессий возросло - ДА
# Подсказка: график будет нагляднее, если трактовать start_month как категориальную порядковую переменную.
#==============================================================================
visualPurpose = pd.DataFrame(index=train_df.index)

# добавим признак start_month
visualPurpose['start_month'] = train_df['time1'].apply(lambda ts: 100 * ts.year + ts.month)#тип int
visualPurpose['target']=y_train
visualPurpose['sessions_count']=1
visualPurpose=pd.DataFrame(data=visualPurpose[visualPurpose.target==1].groupby(['start_month'])['sessions_count'].sum())#группируем данные и считаем количество сессий
visualPurpose.reset_index(level=None, inplace=True)#переносим индекс в колонки

_, axes = plt.subplots(1, 1, sharey=True, figsize=(20,6))#определяем размер графика
sns.stripplot(x="start_month", y="sessions_count", data=visualPurpose, jitter=True, ax=axes,#рисуем график с категориальными переменными
              palette="Set2", size=20, marker="D",
              edgecolor="gray", alpha=.75)
del visualPurpose

#Таким образом, у нас есть иллюстрация и соображения насчет полезности нового признака,
#добавим его в обучающую выборку и проверим качество новой модели:
# добавим новый признак в разреженную матрицу
tmp = full_new_feat[['start_month']].as_matrix()
X_train = csr_matrix(hstack([full_sites_sparse[:idx_split,:], tmp[:idx_split,:]]))

# считаем метрику на валидационной выборке
print(u'start_month (not normal)=',get_auc_lr_valid(X_train, y_train))

#==============================================================================
# Можно дать следующие практические советы:
# * рекомендуется масштабировать признаки, если они находятся в существенно разных шкалах или
# разных единицах измерения (например, население страны указано в единицах, а ВНП страны в триллионах);
# * масштабируйте признаки, если у вас нет оснований/экспертного мнения придавать больший вес каким-либо из них;
# * масштабирование может быть лишним, если диапазоны некоторых ваших признаков отличаются друг от друга,
# но при этом находятся в одной системе единиц (например, доли людей средних лет и старше 80 среди всего населения);
# * если вы хотите получить интерпретируемую модель, то постройте модель без регуляризации и масштабирования
# (скорее всего, ее качество окажется хуже);
# * бинарные переменные (принимают только значения 0 или 1) обычно оставляют без преобразования, (но)
# * если качество модели имеет решающее значение, попробуйте разные варианты и выберите тот, где качество выше.
#==============================================================================
# добавим новый стандартизированный признак в разреженную матрицу
tmp = StandardScaler().fit_transform(full_new_feat[['start_month']])
X_train = csr_matrix(hstack([full_sites_sparse[:idx_split,:], tmp[:idx_split,:]]))

# считаем метрику на валидационной выборке
print(u'start_month (normal)=',get_auc_lr_valid(X_train, y_train))
#start_month (normal)= 0.919699369955

#==============================================================================
# Задание 7: Добавьте в обучающую выборку признак n_unique_sites ,
# количество уникальных сайтов в сессии, и посчитайте, как изменилось качество на отложенной выборке?
# уменьшилось, новый признак лучше не масштабировать - НЕТ
# не изменилось - НЕТ
# уменьшилось, новый признак надо масштабировать - ДА
# я в ступоре и не знаю, надо ли мастшабировать новый признак, а попробовать оба варианта и выбрать лучший не хватает смелости - НЕТ
# Подсказки: воспользуйтесь функцией nunique() из Pandas.
# Не забудьте включить в выборку start_month. Будете ли вы мастшабировать новый признак? Почему?
#==============================================================================
full_new_feat['n_unique_sites']=full_df[sites].apply(lambda x: x.nunique(),axis=1)

tmp = full_new_feat[['n_unique_sites','start_month']].as_matrix()
X_train = csr_matrix(hstack([full_sites_sparse[:idx_split,:], tmp[:idx_split,:]]))

# считаем метрику на валидационной выборке
print(u'n_unique_sites (not normal)=',get_auc_lr_valid(X_train, y_train))


tmp = StandardScaler().fit_transform(full_new_feat[['n_unique_sites','start_month']].as_matrix())
X_train = csr_matrix(hstack([full_sites_sparse[:idx_split,:], tmp[:idx_split,:]]))

# считаем метрику на валидационной выборке
print(u'n_unique_sites (normal)=',get_auc_lr_valid(X_train, y_train))
#n_unique_sites (normal)= 0.91568361762

#==============================================================================
# Задание 8. Добавьте два новых признака: start_hour и morning. Посчитайте метрику, какие из признаков дали прирост?
# Признак start_hour это час в который началась сессия (от 0 до 23), а бинарный признак morning равен 1,
# если сессия началась утром и 0, если сессия началась позже (будем считать, что утро это если start_hour равен 11 или меньше).
# Будете ли вы масштабировать новые признаки? Сделайте предположения и проверьте их на практике.
# ни один из признаков не дал прирост ;(
# start_hour дал прирост, а morning нет
# morning дал прирост, а start_hour почему-то нет
# оба признака дали прирост <-
#==============================================================================
full_new_feat['start_hour']=full_df.time1.map(lambda x: x.hour)
full_new_feat['morning']=full_new_feat.start_hour.map(lambda x: 1 if x<=11 else 0)

def testFeatures(feat,normilize):
    feat.append('start_month')
    tmp=full_new_feat[feat].as_matrix()
    if normilize: tmp = StandardScaler().fit_transform(tmp)
    X_train = csr_matrix(hstack([full_sites_sparse[:idx_split,:], tmp[:idx_split,:]]))

    print('%s (%s)=%s' %(feat,'normal' if normilize else 'general', get_auc_lr_valid(X_train, y_train)))



testFeatures(['start_hour'],False)#start_hour (general)=0.13543115879
testFeatures(['start_hour'],True)#start_hour (normal)=0.957923704404
testFeatures(['morning'],False)#morning (general)=0.817604397635
testFeatures(['morning'],True)#morning (normal)=0.948778419658

testFeatures(['start_hour','morning'],False)#['start_hour', 'morning', 'start_month'] (general)=0.13543115879
testFeatures(['start_hour','morning'],True)#['start_hour', 'morning', 'start_month'] (normal)=0.95915311955


#==============================================================================
# 5. Подбор коэффицициента регуляризации
# Итак, мы ввели признаки, которые улучшают качество нашей модели по сравнению с первым бейслайном.
# Можем ли мы добиться большего значения метрики? После того, как мы сформировали обучающую и тестовую выборки,
# почти всегда имеет смысл подобрать оптимальные гиперпараметры — характеристики модели, которые не изменяются во время обучения.
# Например, на 3 неделе вы проходили решающие деревья, глубина дерева это гиперпараметр, а признак,
# по которому происходит ветвление и его значение — нет. В используемой нами логистической регрессии
# веса каждого признака изменяются, и во время обучения ищутся их оптимальные значения, а
# коэффициент регуляризации остается постоянным. Это тот гиперпараметр, который мы сейчас будем оптимизировать.
# Посчитаем качество на отложенной выборке с коэффициентом регуляризации, который по умолчанию равен 1 (C=1):
#==============================================================================
# формируем обучающую выборку
tmp_scaled = StandardScaler().fit_transform(full_new_feat[['start_month', 'start_hour', 'morning']])
X_train = csr_matrix(hstack([full_sites_sparse[:idx_split,:],
                             tmp_scaled[:idx_split,:]]))

# зафиксируем качество с параметрами по умолчанию
print('Зафиксируем качество с параметрами по умолчанию')
score_C_1 = get_auc_lr_valid(X_train, y_train)
print(score_C_1)

# набор возможных значений С
Cs = np.logspace(-3, 1, 10)#логарифмическая шкала

scores = []
for C in Cs:
    scores.append(get_auc_lr_valid(X_train, y_train, C=C))

#Построим график зависимости метрики от значения коэффициента регуляризации.
#Значение метрики с параметром C по умолчанию отображено горизонтальным пунктиром
_, axes = plt.subplots(1, 1, figsize=(20,6))#определяем размер графика
plt.plot(Cs, scores, 'ro-')
plt.xscale('log')
plt.xlabel('C')
plt.ylabel('AUC-ROC')
plt.title('Подбор коэффициента регуляризации')
# горизонтальная линия -- качество модели с коэффициентом по умолчанию
plt.axhline(y=score_C_1, linewidth=.5, color = 'b', linestyle='dashed')
plt.show()

#Задание 9: при каком коэффициенте регуляризации C модель показывает наивысшее качество?
#0.17 <-
#0.46
#1.29
#3.14

#==============================================================================
# И последнее в этой домашней работе: обучите модель с найденным оптимальным значением
# коэффициента регуляризации (не округляйте до двух знаков как в последнем задании).
# Если вы все сделали правильно и загрузите это решение, то повторите второй бейслайн — 0.93474
# на паблик лидерборде:
#
#==============================================================================
# подготовим данные для обучения и теста
tmp_scaled = StandardScaler().fit_transform(full_new_feat[['start_month', 'start_hour', 'morning']])
X_train = csr_matrix(hstack([full_sites_sparse[:idx_split,:],
                             tmp_scaled[:idx_split,:]]))
X_test = csr_matrix(hstack([full_sites_sparse[idx_split:,:],
                            tmp_scaled[idx_split:,:]]))

# обучим модель на всей выборке с оптимальным коэффициентом регуляризации
lr = LogisticRegression(C=0.17, random_state=17).fit(X_train, y_train)

# сделаем прогноз для тестовой выборки
y_test = lr.predict_proba(X_test)[:, 1]

# запишем его в файл, готовый для сабмита
write_to_submission_file(y_test, 'baseline_2.csv')
