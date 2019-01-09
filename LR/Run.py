from dataload import *
from Logistic import *
def main():
    data_feature,data_label = creat_data()
    x_train, x_test, y_train, y_test = get_train_data(data_feature,data_label)
    model  =LR(200,0.01)
    print("================start training")
    model.fit(x_train,y_train)
    print("=============test my model")
    print("准确率为：",model.score(x_test,y_test))
    print("+++++结束=====")
    print("系数为：",model.weights)

    x_ponits = np.arange(4, 8)
    y_ = -(model.weights[1] * x_ponits + model.weights[0]) / model.weights[2]
    plt.plot(x_ponits, y_)  #打印落在超平面上的点

    # lr_clf.show_graph()
    plt.scatter(data_feature[:50, 0], data_feature[:50, 1], label='0')
    plt.scatter(data_feature[50:, 0], data_feature[50:, 1], label='1')
    plt.show()
if __name__ == '__main__':
    main()

