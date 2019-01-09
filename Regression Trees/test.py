"""
Python strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
注意：该方法只能删除开头或是结尾的字符，不能删除中间部分的字符。

map() 会根据提供的函数对指定序列做映射。
第一个参数 function 以参数序列中的每一个元素调用 function 函数，返回包含每次 function 函数返回值的新列表。
Python 3.x 返回迭代器。 故需要用list将其转化为列表
"""
"""
testlist = [1,2,3,4]
result = map(float,testlist)
print(result[2])
print(list(result))
"""
import numpy as np
"""
dataset = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(dataset[:,1]>3)
print(np.nonzero(dataset[:,1]>3))
print(np.nonzero(dataset[:,1]>3)[0])
"""
X = np.array([[1, 2], [3, 4], [5, 6]])
print(np.var(X, axis=0, keepdims=True))
print(np.var(X, axis=1, keepdims=True))
print(np.var(X))








