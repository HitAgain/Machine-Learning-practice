from scipy import io as spio
import numpy as np
array1 = np.ones((10,1),dtype=np.float32)
print(array1)
print(array1.shape)
array2 = np.ravel(array1)
print(array2)
print(array2.shape)
array3 = np.arange(10)
print(array3)
print(array3.shape)
# np.flatten 返回的是拷贝 改变flatten对象不改变原来的np array   np.ravel则相反



