# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 16:42:12 2017


"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
import pylab as pl
import seaborn as sns


x = np.linspace(0.0001, 2, 1000)
pl.figure()
_vars=[['g',1,'1'], ['b', 0,'0'], ['m', 1/x, '1/x'] , ['y', 1+x,'1+x']]

for color,koef,leg in _vars:
    y=scipy.stats.boxcox(x, lmbda=koef)
    pl.plot(x, y,color,label=leg)

y=np.log(x)
pl.plot(x, y, '.r',label='log')


pl.legend()
pl.xlabel('x')
pl.ylabel('y')
pl.title('График log и boxcox')
pl.show()

print('При каком значении lmbda, выражение np.log(x) == scipy.stats.boxcox(x, lmbda=lmbda) будет истинным. Ответ=0')