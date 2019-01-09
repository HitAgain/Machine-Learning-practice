from dataload import *
from Adaboost import *

def main():
    My_Adaboost = Adaboost(10,0.2)
    x_train,x_test,y_train,y_test = creat_train_data()
    My_Adaboost.fit(x_train,y_train)
    print(My_Adaboost.score(x_test,y_test))
if __name__ == '__main__':
    main()
