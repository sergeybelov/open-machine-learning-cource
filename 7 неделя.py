# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 12:30:55 2017


"""

#==============================================================================
# 1. PCA
# Начнём с того, что импортируем все необходимые модули
#==============================================================================
from matplotlib import pyplot as plt
import seaborn as sns; sns.set(style='white')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn import datasets
from sklearn.model_selection import train_test_split
import sympy
import math
from scipy.linalg import svd
from sklearn import decomposition
from sklearn.metrics.pairwise import euclidean_distances
import seaborn as sns

#Дана игрушечная выборка.
X = np.array([[2., 13.], [1., 3.], [6., 19.],
              [7., 18.], [5., 17.], [4., 9.],
              [5., 22.], [6., 11.], [8., 25.]])

#X=np.array([[-1.5, -0.5],
 #      [-0.5,  0.5],
  #     [ 0.5, -1.5],
   #    [ 1.5,  0. ],
    #   [ 0. ,  1.5]])

plt.scatter(X[:,0], X[:, 1])
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$');

#==============================================================================
# 1. На сколько градусов относительно оси x1 повернут вектор, задающий 1
# главную компоненту в этих данных (на забудьте отмасштабировать выборку)?
#Ответ 45
#==============================================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


plt.scatter(X_scaled[:, 0], X_scaled[:, 1])
plt.plot([-2,2],[0,0], c='black')
plt.plot([0,0],[-2,2], c='black')
#plt.plot([-2,2],[2,-2], c='red');
#plt.plot([-2,2],[1,-1], c='green');

plt.plot([-2,2],[2,-2], c='red');




plt.xlim(-2,2)
plt.ylim(-2,2);
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$');



a, b= sympy.symbols('a b')
formula=[]
z=np.transpose(X_scaled*2)
for i in range(z.shape[1]):
    formula.append('(a*{}+b*{})**2'.format(z.item((0,i)),z.item((1,i))))

formula=sympy.simplify('+'.join(formula))
print('упрощение формулы: ',formula)

print(math.acos(1/math.sqrt(2)))
print('45гр=',math.pi/4)

#t=1
#alpha=beta=1/math.sqrt(2)

#Каковы собственные значения матрицы X^TX , где X – матрица, соответствующая отмасштабированной выборке?
X_scaled.dot(np.array([1./np.sqrt(2), 1./np.sqrt(2)]))


print(' Каковы собственные значения матрицы X^TX , где X – матрица, соответствующая отмасштабированной выборке?',np.linalg.eig(X_scaled.T.dot(X_scaled))[0])
sing=np.linalg.eig(X_scaled.dot(X_scaled.T))[0]

print('В чем смысл двух чисел из прошлого вопроса? эти числа говорят о том, какую часть дисперсии исходных данных объясняют главные компоненты')

lfw_people = datasets.fetch_lfw_people(min_faces_per_person=50,
                resize=0.4, data_home='faces.dat')

print('%d objects, %d features, %d classes' % (lfw_people.data.shape[0],
      lfw_people.data.shape[1], len(lfw_people.target_names)))
#print('\nPersons:')
for name in lfw_people.target_names:
#    print(name)
    fig = plt.figure(figsize=(8, 6))

#Посмотрим на содержимое датасета. Все изображения лежат в массиве lfw_people.images
for i in range(15):
    ax = fig.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
    ax.imshow(lfw_people.images[i], cmap='gray')


#Какое минимальное число компонент PCA необходимо, чтобы объяснить 90% дисперсии
#масштабированных (при помощи StandardScaler) данных?
newVal=[]
for i in range(1560):
    newVal.append(lfw_people.images[i].ravel())

#newVal=lfw_people.images.data


sc_X = StandardScaler().fit_transform(newVal)
pca = decomposition.PCA(svd_solver='randomized',random_state=1).fit(sc_X)

_vars=pca.explained_variance_ratio_
sums=np.cumsum(_vars)
plt.figure(figsize=(10,7))
plt.plot(sums, color='k', lw=3)
plt.xlabel('Number of components')
plt.ylabel('Total explained variance')
plt.xlim(60, 85)
plt.yticks(np.arange(0.8, 1.0, 0.05))
plt.axvline(77, c='b')
plt.axvline(76, c='g')
plt.axhline(0.9, c='r')
plt.show();

n=77
print('Какое минимальное число компонент PCA необходимо, чтобы объяснить 90% дисперсии масштабированных (при помощи StandardScaler) данных?',sums[n-1],'=',n)

#Постройте картинку, на которой изображены первые 30 главных компонент (только не пугайтесь, когда увидите, что получилось).
#Для этого надо эти 30 векторов взять из pca.components_, трансформировать опять по размеру исходных изображений (50 x 37)
#и нарисовать.
#Конкретней: для какой главной компоненты линейная комбинация исходных признаков
#(интенсивностей пикселов), если ее представить как изображение, выглядит, как фотография, ярко освещенная слева.
n=5
newVal30pic=pca.components_[:n].reshape(n,50,37)

for i in range(n):
    fig = plt.figure(figsize=(8, 6))

for i in range(n):
    ax = fig.add_subplot(n, 5, i + 1, xticks=[], yticks=[])
    ax.imshow(newVal30pic[i], cmap='gray')
    ax.title.set_text((i+1))

print('Какая из первых 30 главных компонент сильнее всего "отвечает" за освещенность лица слева? Ответ=2')

#==============================================================================
# Кто меньше всего похож на других людей в выборке, если выделять 2 главные компоненты?
# Для ответа на этот вопрос выделите 2 главные компоненты в масштабированных данных (используйте опять параметр svd_solver='randomized' и random_state=1),
# посчитайте для каждого человека в выборке среднее значение 2 главных компонент всех его фотографий,
# затем из 12 2-мерных точек найдите наиболее удаленную от остальных (по среднему евклидову расстоянию до других точек).
# Можно это делать точно, а можно и на глаз с помощью sklearn.metrics.euclidean_distances и seaborn.heatmap.
#==============================================================================
#Колин Пауэлл Colin Powell
#Джордж Буш George W Bush
#Жак Ширак Jacques Chirac
#Серена Уильямс Serena Williams

def getComp(_index):
    print('Имя: ',lfw_people.target_names[_index],' index=',_index)
    indices=[i for i, x in enumerate(lfw_people.target) if x == _index]

    xx=lfw_people.images[indices]
    print('размеры: {}x{}x{}'.format(xx.shape[0],xx.shape[1],xx.shape[2]))

    newVal=xx.reshape(xx.shape[0],xx.shape[1]*xx.shape[2])
    sc_X = StandardScaler().fit_transform(newVal)
    pca = decomposition.PCA(n_components=2,svd_solver='randomized',random_state=1).fit(sc_X)

    m1=pca.components_[0].mean()
    m2=pca.components_[1].mean()
    print(name,'=',m1,',',m2)
    return m1,m2


#считаем среднее
means=[]
for i in range(12):
    means.append(getComp(i))
means=pd.DataFrame(data=means, index=lfw_people.target_names,dtype=float)

#считаем евклидово расстояние
euc=pd.DataFrame(index=lfw_people.target_names,columns=np.arange(12))
for i in range(12):
    euc.loc[lfw_people.target_names[i]]=euclidean_distances(means.iloc[i].values.reshape(1, -1), means.values)
euc=euc.astype(float)
euc*=100

fig, ax = plt.subplots(figsize=(10,10))# Sample figsize in inches
sns.heatmap(euc, annot=True,  linewidths=.5, ax=ax)

print('-----')
def print_mean(name):
    print(name,'=',euc.loc[name].mean())

print_mean('Colin Powell')
print_mean('George W Bush')
print_mean('Jacques Chirac')
print_mean('Serena Williams')