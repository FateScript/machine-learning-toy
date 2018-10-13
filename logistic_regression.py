#!/usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np


def make_data():
    data = np.array([[1,0.5],[2,1.5],[1.5,2],[0.5,1]])
    label = np.array([1,1,0,0])
    return data,label

data,label = make_data()


def sigmoid(w, x):
    v = np.exp( np.dot(x, w) )
    return v/(1 + v)


class logisticRegresion:

    def __init__(self):
        #self.weights: model's parameters
        #self.p1: probabilty of p(y=1|x;w)
        pass

    def fit_newton(self, x, y):
        # this method is faster than sgd
        # Cross entropy loss: l(w) = -sum(y_i*ln(p1(y|x;w)) + (1-y_i)*ln(p_0(y|x;w)))
        # first derivative of loss : -sum( x_i*(y_i - p_1(x_i;w) ) )
        # second derivative of loss : -sum( x_i*x_i^T * p_1(x_i;w)*(1 - p_1(x_i;w) ) )
        #you can get these two equations by appling chain rule
        self.weights = np.ones(x.shape[1] + 1)
        bias = np.ones((x.shape[0],1))
        x = np.hstack((x,bias))
        while( not self.__convergence(x,y) ):
            #calculate first derivative vector(sgd)
            first_derv_vec = np.dot( y- self.p1, x )
            #calculate second derivative matrix(newton method)
            scale = np.multiply( self.p1,1 - self.p1 )
            x_T = np.multiply(np.transpose(x), scale)
            sec_derv_vec = np.linalg.inv( np.dot(x_T, x) )
            #update parameter
            delta_weights = np.dot(sec_derv_vec, first_derv_vec )
            #delta weights normalizaion
            delta_weights /= abs(delta_weights).max()
            self.weights += delta_weights

    def fit_sgd(self, x, y, learning_rate):
        # first derivative of loss : -sum( x_i*(y_i - p_1(x_i;w) ) )
        self.weights = np.ones(x.shape[1] + 1)
        bias = np.ones((x.shape[0],1))
        #change y = w_T * x + b to y = [w,1]_T * [x,1]
        x = np.hstack((x,bias))
        while( not self.__convergence(x,y) ):
            delta_weights = np.dot(y-self.p1,x)
            delta_weights /= abs(delta_weights).max()
            self.weights += learning_rate * delta_weights
            #if you change += to -= in during update,you will get a overflow error

    def fit_sgd_using_squareLoss(self, x, y, learning_rate):
        #loss function: f = sum( y - sigmoid(w_T*x) )^2
        #first derivative of loss function: -2*(y-p1(x;w))*p1(x;w)*(1-p1(x;w))*x_i
        #you can see that this loss function's gradient is at most 1/4 of cross entropy
        self.weights = np.ones(x.shape[1] + 1)
        bias = np.ones((x.shape[0],1))
        x = np.hstack((x,bias))
        while( not self.__convergence(x,y) ):
            temp = np.multiply( self.p1,(1-self.p1) )
            scale = np.multiply(y-self.p1, temp)
            delta_weights = np.dot(scale, x)
            #print(delta_weights.shape)
            delta_weights /= abs(delta_weights).max()
            self.weights += learning_rate * delta_weights

    def __convergence(self, x, y):
        return (self.__predict_innerData(x) == y).all(axis=0)

    def __predict_innerData(self, x):
        self.p1 = sigmoid(self.weights,x)
        y = 0 + (self.p1 > 0.5)
        return y

    def predict(self, x):
        x = np.hstack( (x, np.ones((x.shape[0],1))) )
        p1 = sigmoid(self.weights,x)
        y = 0 + (p1 > 0.5)
        return y

    def getWeights(self):
        return self.weights


lr = logisticRegresion()
# you can add epoch variable to see how many times it will take to converge
lr.fit_newton(x=data,y=label)
#lr.fit_sgd_using_squareLoss(x=data,y=label,learning_rate=0.5)
#lr.fit_sgd(x=data,y=label,learning_rate=0.5)
print("predict:=", lr.predict(np.array([ [3,4],[2,1] ]) ) )
print("weights=", lr.getWeights())

