import pandas as pd
import numpy as np
import random as rd
import math



class ClusterEngine:
    K=None
    n_iter=None
    path=None
    X=None
    centroids=None
    Y={}

    def __init__(self,K,n_iter,path,attr):
        self.K=K
        self.n_iter = n_iter
        self.X=self.loadData(path,attr)
        print("Loaded Dataset, and hyper-parameters.")

    def loadData(self,path,attr):
        data=pd.read_csv(path)
        data=data.iloc[:,attr].values
        return data

    def initCentroids(self):
        m,n=self.X.shape[0],self.X.shape[1]
        self.centroids=np.array([]).reshape(n,0)
        for i in range(self.K):
            rand=rd.randint(0,m-1)
            self.centroids=np.c_[self.centroids,self.X[rand]]
        print(self.centroids)

    def fit(self):
        C=None
        for iter in range(self.n_iter):
            euclidean_distance=np.array([]).reshape(self.X.shape[0],0)
            for k in range(self.K):
                dist=np.sum((self.X-self.centroids[:,k])**2,axis=1)
                euclidean_distance=np.c_[euclidean_distance,dist]
            C=np.argmin(euclidean_distance,axis=1)+1
            for k in range(self.K):
                self.Y[k+1]=np.array([]).reshape(2,0)
            for i in range(self.X.shape[0]):
                self.Y[C[i]]=np.c_[self.Y[C[i]],self.X[i]]
            for k in range(self.K):
                self.Y[k+1]=self.Y[k+1].T
            for k in range(self.K):
                self.centroids[:,k]=np.mean(self.Y[k+1],axis=0)

class KMeans:

    kmeans=None
    centroids=None
    output=None

    def __init__(self):
        print("Provide path, number of clusters and number of iterations")

    def __init__(self,K,n_iter,path,attr):
        kmeans=ClusterEngine(K,n_iter,path,attr)
        kmeans.initCentroids()
        kmeans.fit()
        self.centroids=kmeans.centroids
        self.output=kmeans.Y
