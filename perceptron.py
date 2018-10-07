#!usr/bin/env python3
# -*- coding:utf-8 -*-

#This code helps you to understand how perceptron(a linear classification model) works

import numpy as np

class perceptron:

    def init(self,x):
        self.weights = np.zeros(x.shape[1])
        self.bias = np.zeros(1)
        self.truepred = np.zeros(x.shape[1])

    def fit(self,x,y,learning_rate):
        # loss function : -sum( y_i*(w*x_i+b) )
        # weight update : -sum( y_i*x_i )
        # bias update   : -sum( y_i )
        self.init(x)
        while(not self.convergence(x,y)):
            index = np.where(self.truepred == False)[0][0]
            self.weights += learning_rate * y[index] * x[index]
            self.bias += learning_rate * y[index]

    def convergence(self,x,y):
        #return wheather or not this model has converged
        self.truepred = self.predict(x) == y
        return self.truepred.all(axis=0)

    def predict(self,x):
        return np.sign(np.dot(self.weights,np.transpose(x)) + self.bias)

    def getParameter(self):
        return self.weights,self.bias

x = np.array([[3,3],[4,3],[1,1]])
y = np.array([1,1,-1])

p = perceptron()
p.fit(x,y,learning_rate=1)
#if you change learning rate,you will see that the parameter scale up/down but the ratio is fixed
#if you want to change the parameter ratio,the only way is to change init function
w,b = p.getParameter()
print("weights:{}\nbias:{}".format(w,b))