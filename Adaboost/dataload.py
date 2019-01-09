import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def creat_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data,columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length','sepal width','petal length','petal width','label']
    data = np.array(df.iloc[:100,[0,1,-1]])
    for i in range(len(data)):
        if data[i,-1] == 0:
            data[i,-1] = -1
    return data[:,:2],data[:,-1]
def creat_train_data():
    X,Y = creat_data()
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)
    return x_train,x_test,y_train,y_test
