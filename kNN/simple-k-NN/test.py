import numpy as np
testdata = [2,3]
traindatasize = 3
#result = np.tile(testdata,(3,1))
#print(result - [[1,1],[1,1],[1,1]])
"""
下面的方式代替了np.tile() 的效果
"""
array1 = np.array(testdata)
for i in range(traindatasize-1):
    array1 = np.vstack((array1,testdata))
print(array1)
