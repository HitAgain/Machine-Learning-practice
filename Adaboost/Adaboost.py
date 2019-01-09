import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
class Adaboost():
    def __init__(self,num_estimator=50,learning_rate = 1.0):
        self.clf_num = num_estimator   #基分类器的个数
        self.learning_rate = learning_rate

    def init_args(self,features,labels):
        self.X = features
        self.Y = labels
        self.M,self.N = features.shape  #样本个数  特征的维度

        self.clf_sets = []   #记录每个基分类器

        self.alpha = []  #记录每个基分类器的权重系数
        self.weights = [1.0/self.M] * self.M # 初始化每个样本的权重相同
    """
    _G函数作用是找到特征再某一维度上的作为分类器的最佳分界值
    """
    def _G(self,features,labels,weights):  #输入时单一维度的feature shape = [self.M,1]
        m = len(features)
        error = 100000.0  # 无穷大
        best_v = 0.0
        # 单维features
        features_min = min(features)
        features_max = max(features)
        n_step = (features_max - features_min + self.learning_rate) // self.learning_rate
        direct, compare_array = None, None

        for i in range(1, int(n_step)):
            v = features_min + self.learning_rate * i

            if v not in features:
                # 误分类计算  以临界值做二分类 两种情况 大于临界值 为1 或者为 -1
                compare_array_positive = np.array([1 if features[k] > v else -1 for k in range(m)])
                weight_error_positive = sum([weights[k] for k in range(m) if compare_array_positive[k] != labels[k]])

                compare_array_nagetive = np.array([-1 if features[k] > v else 1 for k in range(m)])
                weight_error_nagetive = sum([weights[k] for k in range(m) if compare_array_nagetive[k] != labels[k]])

                if weight_error_positive < weight_error_nagetive:
                    weight_error = weight_error_positive
                    _compare_array = compare_array_positive
                    direct = 'positive'
                else:
                    weight_error = weight_error_nagetive
                    _compare_array = compare_array_nagetive
                    direct = 'nagetive'

                # print('v:{} error:{}'.format(v, weight_error))
                if weight_error < error:
                    error = weight_error
                    compare_array = _compare_array
                    best_v = v
        return best_v, direct, error, compare_array    #返回当前特征最佳分割值，分类准则，误差，分类结果

    def _alpha(self, error):
        return 0.5 * np.log((1 - error) / error)

        # 规范化因子

    def _Z(self, weights, a, clf):
        return sum([weights[i] * np.exp(-1 * a * self.Y[i] * clf[i]) for i in range(self.M)])
        # 权值更新
    def _w(self, a, clf, Z):
        for i in range(self.M):
            self.weights[i] = self.weights[i] * np.exp(-1 * a * self.Y[i] * clf[i]) / Z

    def _f(self, alpha, clf_sets):
        pass

    def G(self, x, v, direct):
        if direct == 'positive':
            return 1 if x > v else -1
        else:
            return -1 if x > v else 1

    def fit(self, X, y):
        self.init_args(X, y)

        for epoch in range(self.clf_num): #每个epoch里面训练得到一个简单分类器
            best_clf_error, best_v, clf_result = 100000, None, None
            # 根据特征维度, 选择误差最小的
            for j in range(self.N):  #遍历所有特征
                features = self.X[:, j]
                # 分类阈值，分类误差，分类结果
                v, direct, error, compare_array = self._G(features, self.Y, self.weights)

                if error < best_clf_error:
                    best_clf_error = error
                    best_v = v
                    final_direct = direct
                    clf_result = compare_array
                    axis = j

                # print('epoch:{}/{} feature:{} error:{} v:{}'.format(epoch, self.clf_num, j, error, best_v))
                if best_clf_error == 0:
                    break

            # 计算G(x)系数a
            a = self._alpha(best_clf_error)
            self.alpha.append(a)
            # 记录分类器
            self.clf_sets.append((axis, best_v, final_direct))  # 表示该分类器以特征的第axis维为分类特征，以best_v为分界值，final_direct为分类准则
            # 规范化因子
            Z = self._Z(self.weights, a, clf_result)
            # 权值更新
            self._w(a, clf_result, Z)
            print('分类器:{}/{} 误差:{:.3f} 分界值:{} 分界准则:{} 特征轴:{:.5f}'.format(epoch+1, self.clf_num, error, best_v, final_direct, axis))
        print('分类器的权重:{}'.format(self.weights))
        print('\n')

    def predict(self, feature):
        result = 0.0
        for i in range(len(self.clf_sets)):
            axis, clf_v, direct = self.clf_sets[i]
            f_input = feature[axis]
            result += self.alpha[i] * self.G(f_input, clf_v, direct)  #多分类器加权投票
        # sign
        return 1 if result > 0 else -1

    def score(self, X_test, y_test):
        right_count = 0
        for i in range(len(X_test)):
            feature = X_test[i]
            if self.predict(feature) == y_test[i]:
                right_count += 1
        return right_count / len(X_test)







