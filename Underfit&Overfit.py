#! /usr/bin/env python3
# -*- coding:utf-8 -*-
#This code gives a simple example of how underfit & overfit works
#We use least square models to fit data and the best parameters is given with Linear Algebra method
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2 + 2*x + 1

def loss(real_label,train_label):
    dif = np.array([x-y for x,y in zip(real_label,train_label)])
    return sum(dif**2)

class LeastSquareModel:

    def __init__(self,fit_size):
        self.size = fit_size

    def fit(self,sample,label):
        A = np.mat([1]*len(label))
        b = np.mat(label).T
        for i in range(self.size-1):
            temp = sample**(i+1)
            A= np.vstack( (A,np.mat(temp)) )
        self.weight = ((A*A.T).I)*(A*b)

    def get_label(self,x):
        label = [0]*len(x)
        for i in range(self.size):
            label += self.weight[i]*x**i
        return label.tolist()[0]

    def get_weight(self):
        return self.weight

sampleNum = 10
ori_data = np.array(list(range(-10,10,2)))

overfit_size,real_size,underfit_size = [6,3,2] # this list decides the model complexity
data = f(ori_data) + np.random.randint(-3,3,sampleNum)

overfit_model = LeastSquareModel(overfit_size)
real_model = LeastSquareModel(real_size)
underfit_model = LeastSquareModel(underfit_size)

overfit_model.fit(ori_data,data)
real_model.fit(ori_data,data)
underfit_model.fit(ori_data,data)

x = np.linspace(-10,10,sampleNum*100)
y = f(x)
p1 = plt.subplot(221)
p1.set_title("original data")
p1.plot(x,y,color='r')
p1.plot(ori_data,data,'x')
print("original loss = ",loss(data,f(ori_data)) )

p2 = plt.subplot(222)
p2.set_title('overfit')
p2.plot(x,overfit_model.get_label(x),color='g')
p2.plot(ori_data,data,'x')
print( "overfit loss = ",loss(data,overfit_model.get_label(ori_data)) )

p3 = plt.subplot(223)
p3.set_title('normal fit')
p3.plot(x,real_model.get_label(x),color='b')
p3.plot(ori_data,data,'x')
print("normal loss = ",loss(data,real_model.get_label(ori_data)))

p4 = plt.subplot(224)
p4.set_title('underfit')
p4.plot(x,underfit_model.get_label(x),color='b')
p4.plot(ori_data,data,'x')
print("underfit loss = ",loss(data,underfit_model.get_label(ori_data)))

plt.show()