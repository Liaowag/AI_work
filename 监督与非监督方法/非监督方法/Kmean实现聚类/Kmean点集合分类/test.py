#coding: utf-8
import random
from numpy import *
import numpy as np
import Kmean
import matplotlib
import matplotlib.pyplot as plt

points = [[1, 2], [2, 1], [3, 1], [5, 4], [5, 5], [6, 5], [7, 9], [99, 96], [94, 91], [92, 89],[87, 34],[45, 78]]
algo =Kmean.Kmean()
mu, clusters_points = algo.get_clusters()

length=[]
point=[]
for c in clusters_points:
    length.append(len(c))
    for b in c:
        point.append(b)
point=array(point)
i=1
label=[]
plt.subplot(212)
for num in length:
    label += list(i*ones(num))
    i += 1
label = array(label)
plt.scatter(point[:, 1], point[:, 0], 15.0 * label, 15.0 * label)
plt.show()

print (mu)
print (clusters_points)
