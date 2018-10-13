#!/usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
import math
import matplotlib.pyplot as plt
#from .svd import svd

def SVD(M, use_np=True):
    if use_np:
        return np.linalg.svd(M)
    else:
        pass

class LDA:

    def __init__(self):
        pass

    def getSw(self, x0, x1):
        x0_cov = np.cov(np.transpose(x0))
        x1_cov = np.cov(np.transpose(x1))
        return x0_cov + x1_cov

    def getSb(self):
        diff_vec = np.matrix(self.x0_mean - self.x1_mean)
        return diff_vec.T * diff_vec

    def inverse(self, S):
        S = np.matrix(S)
        u,s,v = SVD(S)
        return np.array( (np.multiply(1/s,v.T)) * u.T )

    def fit(self, x0, x1):
        #LDA maximize generalized Rayleigh quotient
        #generalized Rayleigh quotient: w.T * Sb * w / w.T * Sw *w
        #Sw = sigma0 + sigma1 while sigma0 is the covariance matrix of class0,sigma1 similarly
        #Sw is also named within-class scatter matrix
        #Sb = (mu_0 - mu_1)*(mu_0 - mu_1).T, named between-class scatter matrix
        #mu_0 and mu1 are the mean vectors of class0 and class1
        self.x0_mean = np.mean(x0, axis=0)
        self.x1_mean = np.mean(x1, axis=0)
        self.Sw = self.getSw(x0, x1)
        self.Sb = self.getSb()
        w = np.dot( self.inverse(self.Sw), self.x0_mean - self.x1_mean )
        # w only decides the direction of the line,so we need to normalize it
        #self.weights = w/abs(w).min()
        self.weights = -w/w[-1]

    def getWeights(self):
        return self.weights

    def showDifference(self):
        weights = np.matrix(self.weights)
        diff = self.x0_mean - self.x1_mean
        print("best result:", weights * self.Sb * weights.T / (weights*self.Sw*weights.T))
        temp = np.matrix([2,-1])
        print("compared result:",temp* self.Sb * temp.T / (temp*self.Sw*temp.T))


x0 = np.array([[0,1],[1,3],[2,2]])
x1 = np.array([[6,2],[4,5],[5,8]])
model = LDA()
model.fit(x0,x1)
model.showDifference()
# show how the line and data look like (2dim data required)
plt.scatter(x0[:,0],x0[:,1],marker='x')
print("weights:", model.getWeights())
plt.scatter(x1[:,0],x1[:,1],marker='+')
x = np.linspace(0,4,1000)
y = model.getWeights()[0]*x
plt.plot(x,y)
plt.show()

