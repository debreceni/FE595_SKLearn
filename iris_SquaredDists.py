# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 21:12:08 2019

@author: dave
"""

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd


def buildDistances():
    iris = load_iris()
    X = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])
    
    sqdist = []
    for i in range(1,15):       
        model = KMeans(n_clusters=i).fit(X)
        sqdist.append(model.inertia_)
        
    return sqdist



def plot_distances(dists):
    
    plt.plot([i for i in range(1,len(dists)+1)],dists)
    plt.xticks([i for i in range(1,len(dists)+1)])
    plt.suptitle('Squared Distances vs. # of Clusters', color= 'blue', fontsize=16)
    plt.xlabel('Number of Clusters', fontsize=12, color='blue')
    plt.ylabel('Squared Distances', fontsize=12, color='blue')
    
    plt.show()
    
    
if __name__ == '__main__':
    dist = buildDistances()
    plot_distances(dist)