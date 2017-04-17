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
import matplotlib.cm as cm
from sklearn.datasets import load_boston
from sklearn.cluster import AffinityPropagation, SpectralClustering
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist


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
newVal=lfw_people.images.reshape(lfw_people.images.shape[0],lfw_people.images.shape[1]*lfw_people.images.shape[2])

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

#def getComp(_index):
#    print('Имя: ',lfw_people.target_names[_index],' index=',_index)
#    indices=[i for i, x in enumerate(lfw_people.target) if x == _index]
#
#    xx=lfw_people.images[indices]
#    print('размеры: {}x{}x{}'.format(xx.shape[0],xx.shape[1],xx.shape[2]))
#
#    newVal=xx.reshape(xx.shape[0],xx.shape[1]*xx.shape[2])
#    sc_X = StandardScaler().fit_transform(newVal)
#    pca = decomposition.PCA(n_components=2,svd_solver='randomized',random_state=1).fit(sc_X)
#
#    m1=pca.components_[0].mean()
#    m2=pca.components_[1].mean()
#    print(name,'=',m1,',',m2)
#    return m1,m2

pca = pd.DataFrame(data=decomposition.PCA(n_components=2,svd_solver='randomized',random_state=1).fit_transform(sc_X))
pca[3]=lfw_people.target

means=pca.groupby([3])[0,1].mean()
means.index=lfw_people.target_names


#считаем евклидово расстояние
euc=pd.DataFrame(index=lfw_people.target_names,columns=np.arange(12))
for i in range(12):
    euc.loc[lfw_people.target_names[i]]=euclidean_distances(means.iloc[i].values.reshape(1, -1), means.values)
euc=euc.astype(float)


fig, ax = plt.subplots(figsize=(10,10))# Sample figsize in inches
sns.heatmap(euc, annot=True,  linewidths=.5, ax=ax)

print('-----')
def print_mean(name):
    print(name,'=',euc.loc[name].mean())

print_mean('Colin Powell')
print_mean('George W Bush')
print_mean('Jacques Chirac')
print_mean('Serena Williams')

#Какому человеку из набора данных lfw_people соответствуют два выброса в правом верхнем углу проекции
#t-SNE с параметрами n_components=2 и random_state=1
print('-----')

newVal=lfw_people.images.reshape(lfw_people.images.shape[0],lfw_people.images.shape[1]*lfw_people.images.shape[2])
sc_X = StandardScaler().fit_transform(newVal)

tsne = TSNE(n_components=2, random_state=1)
X_tsne = tsne.fit_transform(sc_X)

plt.figure(figsize=(12,10))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=lfw_people.target,
            cmap=plt.cm.get_cmap('nipy_spectral', 12), alpha=1, s=60)
plt.colorbar()

print('Какому человеку из набора данных lfw_people соответствуют два выброса в правом верхнем углу проекции t-SNE с параметрами n_components=2 и random_state=1? Serena Williams')
#plt.colorbar(sc)

#Каким будет оптимальное число кластеров для датасета с ценами на жильё, если оценивать его с помощью метода локтя?
#Используйте в kMeans random_state=1, данные не масштабируйте.
#Найдем с помощью метода локтя (см. 7 статью курса) оптимальное число кластеров, которое стоит задать алгоритму kMeans в качестве гиперпараметра.
#Каким будет оптимальное число кластеров для датасета с ценами на жильё,
#если оценивать его с помощью метода локтя? Используйте в kMeans random_state=1, данные не масштабируйте.
boston = load_boston()
X = boston.data


inertia = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=1).fit(X)
    inertia.append(np.sqrt(kmeans.inertia_))

plt.figure(figsize=(12,10))
plt.plot(range(2, 10), inertia, marker='s');
plt.xlabel('$k$')
plt.ylabel('$J(C_k)$');

print('Каким будет оптимальное число кластеров для датасета с ценами на жильё, если оценивать его с помощью метода локтя? - 4')

#9. Выберите все верные утверждения


print('9. Выберите все верные утверждения')

def testClusters(X,y,n):
    algorithms = []
    algorithms.append(KMeans(n_clusters=n, random_state=1))
    algorithms.append(AffinityPropagation())
    algorithms.append(SpectralClustering(n_clusters=n, random_state=1,
                                     affinity='nearest_neighbors'))
    algorithms.append(AgglomerativeClustering(n_clusters=n))

    data = []
    for algo in algorithms:
        algo.fit(X)
        data.append(({
                'ARI': metrics.adjusted_rand_score(y, algo.labels_),
                'AMI': metrics.adjusted_mutual_info_score(y, algo.labels_),
                'Homogenity': metrics.homogeneity_score(y, algo.labels_),
                'Completeness': metrics.completeness_score(y, algo.labels_),
                'V-measure': metrics.v_measure_score(y, algo.labels_),
                'Silhouette': metrics.silhouette_score(X, algo.labels_)}))

    results = pd.DataFrame(data=data, columns=['ARI', 'AMI', 'Homogenity',
                                               'Completeness', 'V-measure',
                                               'Silhouette'],
                           index=['K-means', 'Affinity',
                                  'Spectral', 'Agglomerative'])
    print(results)


print('general')
testClusters(sc_X, lfw_people.target,12)

target=np.zeros(lfw_people.target.shape[0])
indices=[i for i, x in enumerate(lfw_people.target) if x == 10]
target[indices]=1

print('Serena Williams')
testClusters(sc_X, target,2)

print('Affinity Propagation сработала лучше спектральной кластеризации по всем метрикам качества')
print('Если выделять только 2 кластера, а результаты кластеризации сравнивать с бинарным вектором, Серена Уильямс это или нет, то в целом алгоритмы справляются лучше, некоторые метрики превышают значение в 66%')

#Возьмите полученные раннее координаты 12 "средних" изображений людей.
#Постройте для них дендрограмму. Используйте scipy.cluster.hierarchy
#и scipy.spatial.distance.pdist, параметры возьмите такие же, как в соответствующем примере в статье.

#meandist=wholeMean#wholeMean.values
#meandist=pd.DataFrame(means.mean(axis=1))
#meandist[1]=list(range(12))

#meandist=means.mean(axis=1)
distance_mat = pdist(means.values) # pdist посчитает нам верхний треугольник матрицы попарных расстояний

Z = hierarchy.linkage(distance_mat, 'single') # linkage — реализация агломеративного алгоритма
plt.figure(figsize=(7, 7))
dn = hierarchy.dendrogram(Z, color_threshold=0.1)

print('Какому человеку соответствует точка, объединившаяся с другими при построении дендрограммы предпоследней?',lfw_people.target_names[7])