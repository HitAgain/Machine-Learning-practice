from math import exp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
def creat_data():
    iris = load_iris()
    origin_data  = pd.DataFrame(iris.data, columns=iris.feature_names)
    origin_data['label'] = iris.target
    origin_data.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(origin_data.iloc[:100, [0, 1, -1]])
    return data[:,:2], data[:,-1]
def get_train_data(x,y):
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
    return x_train,x_test,y_train,y_test
