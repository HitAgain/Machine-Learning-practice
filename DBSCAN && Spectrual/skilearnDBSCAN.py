# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 15:53:02 2018

@author: Again
"""
###  利用skilearn的做DBSCAN聚类  
###在scikit-learn中，DBSCAN算法类为sklearn.cluster.DBSCAN。要熟练的掌握用DBSCAN类来聚类，除了对DBSCAN本身的原理有较深的理解以外，还要对最近邻的思想有一定的理解。集合这两者，就可以玩转DBSCAN了。
"""
DBSCAN类的重要参数也分为两类，一类是DBSCAN算法本身的参数，一类是最近邻度量的参数，下面我们对这些参数做一个总结。

1）eps： DBSCAN算法参数，即我们的ϵ-邻域的距离阈值，和样本距离超过ϵ的样本点不在ϵ-邻域内。
  默认值是0.5.一般需要通过在多组值里面选择一个合适的阈值。eps过大，则更多的点会落在核心对象的ϵ-邻域，
  此时我们的类别数可能会减少， 本来不应该是一类的样本也会被划为一类。反之则类别数可能会增大，本来是一类的样本却被划分开。

2）min_samples： DBSCAN算法参数，即样本点要成为核心对象所需要的ϵ-邻域的样本数阈值。默认值是5. 
一般需要通过在多组值里面选择一个合适的阈值。通常和eps一起调参。在eps一定的情况下，
min_samples过大，则核心对象会过少，此时簇内部分本来是一类的样本可能会被标为噪音点，类别数也会变多。
反之min_samples过小的话，则会产生大量的核心对象，可能会导致类别数过少。


3）metric：最近邻距离度量参数。可以使用的距离度量较多，一般来说DBSCAN使用默认的欧式距离（即p=2的闵可夫斯基距离）就可以满足我们的需求。
可以使用的距离度量参数有：
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
X1, y1=datasets.make_circles(n_samples=5000, factor=.6,
                                      noise=.05)
X2, y2 = datasets.make_blobs(n_samples=1000, n_features=2, centers=[[1.2,1.2]], cluster_std=[[.1]],
               random_state=9)

X = np.concatenate((X1, X2))   ##拼接函数 默认时行拼接 即上下拼
plt.scatter(X[:, 0], X[:, 1], marker='o')
plt.show()

"""
效果图可以看出K-Means对非凸的数据集表现不好
"""
from sklearn.cluster import KMeans  
y_pred = KMeans(n_clusters=4, random_state=9).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()


from sklearn.cluster import DBSCAN
#######
#y_pred = DBSCAN().fit_predict(X)
#plt.scatter(X[:, 0], X[:, 1], c=y_pred)
#plt.show()
#######
y_pred = DBSCAN(eps = 0.1).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()

y_pred = DBSCAN(eps = 0.1, min_samples = 10).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()
"""
和传统的K-Means算法相比，DBSCAN最大的不同就是不需要输入类别数k，当然它最大的优势是可以发现任意形状的聚类簇，而不是像K-Means，一般仅仅使用于凸的样本集聚类。同时它在聚类的同时还可以找出异常点，这点和BIRCH算法类似。

　　　　那么我们什么时候需要用DBSCAN来聚类呢？一般来说，如果数据集是稠密的，并且数据集不是凸的，那么用DBSCAN会比K-Means聚类效果好很多。如果数据集不是稠密的，则不推荐用DBSCAN来聚类。

　　　　下面对DBSCAN算法的优缺点做一个总结。

　　　　DBSCAN的主要优点有：

　　　　1） 可以对任意形状的稠密数据集进行聚类，相对的，K-Means之类的聚类算法一般只适用于凸数据集。

　　　　2） 可以在聚类的同时发现异常点，对数据集中的异常点不敏感。

　　　　3） 聚类结果没有偏倚，相对的，K-Means之类的聚类算法初始值对聚类结果有很大影响。

　　　　DBSCAN的主要缺点有：

　　　　1）如果样本集的密度不均匀、聚类间距差相差很大时，聚类质量较差，这时用DBSCAN聚类一般不适合。

　　　　2） 如果样本集较大时，聚类收敛时间较长，此时可以对搜索最近邻时建立的KD树或者球树进行规模限制来改进。

　　　　3） 调参相对于传统的K-Means之类的聚类算法稍复杂，主要需要对距离阈值ϵ，邻域样本数阈值MinPts联合调参，不同的参数组合对最后的聚类效果有较大影响。
"""









