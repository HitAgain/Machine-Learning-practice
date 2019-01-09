import numpy as np
import math

"""
以三硬币模型作为最简单的模拟
"""
class EM:
    def __init__(self, prob):
        self.pro_A, self.pro_B, self.pro_C = prob

    # e_step
    def pmf(self, i,data):
        pro_1 = self.pro_A * math.pow(self.pro_B, data[i]) * math.pow((1 - self.pro_B), 1 - data[i])
        pro_2 = (1 - self.pro_A) * math.pow(self.pro_C, data[i]) * math.pow((1 - self.pro_C), 1 - data[i])
        return pro_1 / (pro_1 + pro_2)

    # m_step
    def fit(self, data):
        count = len(data)
        print('init prob:{}, {}, {}'.format(self.pro_A, self.pro_B, self.pro_C))
        for d in range(count):
            _ = yield
            _pmf = [self.pmf(k,data) for k in range(count)]   # 观测变量服从B分布的后验概率列表
            ## 更新
            pro_A = 1 / count * sum(_pmf)
            pro_B = sum([_pmf[k] * data[k] for k in range(count)]) / sum([_pmf[k] for k in range(count)])
            pro_C = sum([(1 - _pmf[k]) * data[k] for k in range(count)]) / sum([(1 - _pmf[k]) for k in range(count)])
            ## 打印
            print('{}/{}  pro_a:{:.3f}, pro_b:{:.3f}, pro_c:{:.3f}'.format(d + 1, count, pro_A, pro_B, pro_C))
            self.pro_A = pro_A
            self.pro_B = pro_B
            self.pro_C = pro_C
def  main():
    data = [1, 1, 0, 1, 0, 0, 1, 0, 1, 1]
    em = EM(prob=[0.5, 0.5, 0.5])
    f = em.fit(data)
    next(f)
    print(f.send(1))
    print(f.send(2))
if __name__ == '__main__':
    main()
