import numpy as np
import scipy as sp
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
def real_fun(x):
    return np.sin(2*np.pi*x)

def fit_fun(p,x):
    f = np.poly1d(p)
    return f(x)

def res_fun(p,x,y_real):
    ret = fit_fun(p, x)-y_real
    return ret
def res_fun_addregular(p,x,y_real):
    regularization = 0.0001
    ret = fit_fun(p,x)- y_real
    ret = np.append(ret,np.sqrt(0.5*regularization*np.square(p)))
    return ret
def fitting(M,x,y,mode):
    if mode == "regular":
        p_init = np.random.rand(M+1) #随机初始化多项式参数  参数比最高次数多1
        p_lsq = leastsq(res_fun_addregular,p_init,args=(x,y))
        print("多项式的参数", p_lsq[0])
        return p_lsq[0]
    else:
        p_init = np.random.rand(M+1)
        p_lsq = leastsq(res_fun,p_init,args=(x,y))
        print("多项式的参数",p_lsq[0])
        return p_lsq[0]
def inference(testpoints,trained_parameters,x,y):
    y_real = real_fun(testpoints)
    y_predict = fit_fun(trained_parameters,testpoints)
    plt.plot(testpoints, y_real,label="real")
    plt.plot(testpoints, y_predict,label="fitted")
    plt.plot(x, y,'bo',label='noise')
    plt.show()
def main():
    x = np.linspace(0,1,10)
    testpoints = np.linspace(0,1,1000)
    y_ = real_fun(x)
    y = [np.random.normal(0, 0.1)+Y for Y in y_]
    parameters = fitting(3,x,y,mode="regular")
    inference(testpoints,parameters,x,y)
if __name__ == '__main__':
    main()

