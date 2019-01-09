from dataload import *
from model import Model
import numpy as np
def main():
    mydata = load_data()
    mytrain_data = set_data_frame(mydata)
    print("========数据可视化===========")
    data_show(mytrain_data)
    print("========获得训练数据==========")
    train_features,train_labels,feature_dim = get_train_data(mytrain_data)
    print("=========初始化模型===========")
    my_model = Model(feature_dim)
    flag= my_model.fit(train_features,train_labels)
    if flag:
        print("============训练完毕==========")
    print("============测试模型===========")
    test_points = np.linspace(4,7,10)
    y_ = -(my_model.w[0]*test_points + my_model.b)/my_model.w[1]  #绘制落在超平面的点 即满足带入sign函数等于0的x,y的值
    print("===========绘图============")
    plt.plot(test_points, y_)
    plt.plot(mytrain_data[:50]['sepal_length'], mytrain_data[:50]['sepal_width'], 'bo', color='blue', label='0')
    plt.plot(mytrain_data[50:100]['sepal_length'], mytrain_data[50:100]['sepal_width'], 'bo', color='orange', label='1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.show()
    print("===========获得模型参数=============")
    w,b = my_model.get_config()
    print(w,b)
    print("===========All Finished==============")
if __name__ == '__main__':
    main()