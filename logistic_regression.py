#!/usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import math

def make_dataset(num = 100):
    data = np.ones(200)
    label = np.hstack( (np.ones(num),np.zeros(num)) )
    return data,label

def make_data():
    data = np.array([[1,0.5],[2,1.5],[1.5,2],[0.5,1]])
    label = np.array([1,1,0,0])
    return data,label

data,label = make_data()
print(data)
print(label)

#plt.scatter(data[:,1],label)
#plt.show()
#print(data[0])
#print(label[0])


class logisticRegresion:

    def __init__(self):
        pass

    def fit_newton(self, x, y):
        # first derivative of loss : -sum( x_i*(y_i - p_1(x_i;w) ) )
        # second derivative of loss : -sum( x_i*x_i^T * p_1(x_i;w)*(1 - p_1(x_i;w) ) )
        bias = np.ones((x.shape[0],1))
        self.weight = np.ones(x.shape[1] + 1)
        x = np.hstack((x,bias))
        while( not self.convergence(x,y) ):
            beta_x = np.exp(np.dot(x,self.weight))
            p1 = beta_x/( 1 + beta_x)
            #calculate first derivative vector(sgd)
            first_derv_vec = np.dot( y- p1, x )
            #calculate second derivative vector(newton method)
            scale = np.multiply( p1, 1-p1 )
            x_T = np.multiply(np.transpose(x), scale)
            sec_derv_vec = np.linalg.inv( np.dot(x_T, x) )
            delta_weight = np.dot(sec_derv_vec, first_derv_vec )
            norm = abs(delta_weight).max()
            delta_weight /= abs()
            print("delta_weight",delta_weight)
            print("first derivative",first_derv_vec)
            self.weight += delta_weight

    def fit_sgd(self, x, y, learning_rate):
        # first derivative of loss : -sum( x_i*(y_i - p_1(x_i;w) ) )
        bias = np.ones((x.shape[0],1))
        self.weight = np.ones(x.shape[1] + 1)
        x = np.hstack((x,bias))
        while( not self.convergence(x,y) ):
            #beta_x = np.exp(np.dot(x,self.weight))
            #p1 = beta_x/( 1 + beta_x)
            delta_weight = np.dot(y-self.p1,x)
            norm = abs(delta_weight).max()
            delta_weight /= norm
            self.weight += learning_rate * delta_weight
            #if you change += to -= in during update,you will get a overflow error

    def convergence(self, x, y):
        return (self.predict(x) == y).all(axis=0)

    def predict(self, x):
        v = np.exp( np.dot(x, self.weight) )
        self.p1 = v/(1 + v)
        y = 0 + (self.p1 > 0.5)
        return y

    def predict_out(self, x):
        x = np.hstack( (x, np.ones((x.shape[0],1))) )
        v = np.exp( np.dot(x, self.weight) )
        p1 = v/(1 + v)
        print(p1)
        y = 0 + (p1 > 0.5)
        return y

    def getWeight(self):
        return self.weight


lr = logisticRegresion()
#lr.fit_sgd(x=data,y=label,learning_rate=0.1)
lr.fit_newton(x=data,y=label)
print("predict:=", lr.predict_out(np.array([ [3,4],[2,1] ]) ) )
print("weights=", lr.getWeight())

