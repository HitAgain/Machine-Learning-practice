from math import exp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
class LR():
    def __init__(self,max_iterator=200,learning_rate = 0.01):
        self.max_iterator = max_iterator
        self.learning_rate = learning_rate
    def sigmoid(self,x):
        return 1/(1+exp(-x))
    def data_add_one_dim(self,data_feature): #回归系数比特征数多一个，即常数项 故原来的特征向量统统在前面添加一个1
        #data_mat = []
        pad = np.ones((len(data_feature),1),dtype=np.float32) #直接用numpy的水平拼接
        data_mat = np.hstack((pad,data_feature))
        #for d in range(len(data_feature)):
         #   data_mat.append([1.0,*data_feature[d]])
        return data_mat
    def fit(self,x,y):
        data_feature = self.data_add_one_dim(x)
        data_label = y
        self.weights = np.zeros((len(data_feature[0]),1),dtype=np.float32)
        for ite in range(self.max_iterator):
            for i in range(len(data_feature)):
                result = self.sigmoid(np.dot(data_feature[i],self.weights))
                error = data_label[i] - result
                self.weights += self.learning_rate * error * np.transpose([data_feature[i]])
        print('LogisticRegression Model(learning_rate={},max_iter={})'.format(self.learning_rate, self.max_iterator))
    def score(self, X_test, y_test):
        right = 0
        X_test = self.data_add_one_dim(X_test)
        for x, y in zip(X_test, y_test):
            result = np.dot(x, self.weights)
            if (result > 0 and y == 1) or (result < 0 and y == 0):
                right += 1
        return right / len(X_test)



