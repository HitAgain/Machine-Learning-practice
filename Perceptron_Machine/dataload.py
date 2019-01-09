import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
def load_data():
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data['label'] = iris.target
    print(data)
    print("data loaded finished")
    return data
def set_data_frame(data):
    data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label']
    print(data.label.value_counts())
    return data


def data_show(data):
    plt.scatter(data[:50]['sepal_length'], data[:50]['sepal_width'],label = '0')
    plt.scatter(data[50:100]['sepal_length'], data[50:100]['sepal_width'],label='1')
    plt.xlabel('sepal_length')
    plt.ylabel('sepal_width')
    plt.show()
def get_train_data(data):
    data = np.array(data.iloc[:100,[0,1,-1]])
    train_features = data[:, :2]
    y = data[:, -1]
    train_label = np.array([1 if i == 1 else -1 for i in y]) # label = 0 --> label = -1
    feature_dim = len(train_features[0])
    print(train_features)
    print(train_label)
    print("特征维数为：",feature_dim)
    return train_features, train_label,feature_dim

"""
def main():
    data = load_data()
    train_data = set_data_frame(data)
    data_show(train_data)
    features, labels = get_train_data(train_data)
    print(features, labels)
if __name__ == '__main__':
    main()
"""

