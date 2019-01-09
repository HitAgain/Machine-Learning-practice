
import numpy as np
import operator
def createDataSet():
	#四组二维特征
	group = np.array([[1,101],[5,89],[108,5],[115,8]])
	#四组特征的标签
	labels = ['爱情片','爱情片','动作片','动作片']
	return group, labels

"""
函数说明:kNN算法,分类器

Parameters:
	inX - 用于分类的数据(测试集)
	dataSet - 用于训练的数据(训练集)
	labes - 分类标签
	k - kNN算法参数,选择距离最小的k个点
Returns:
	sortedClassCount[0][0] - 分类结果
"""
def classify0(inX, dataSet, labels, k):
	# 一个待测数据需要和每一个训练数据作差  所以行向上要copy  len（dataset）-1 次。
	dataSetSize = len(dataSet)
	array_inX = np.array(inX)
	for i in range(dataSetSize - 1):  ##这里注意 一定减1
		array_inX= np.vstack((array_inX, inX))
	#二维特征相减后平方
	diffMat = array_inX - dataSet
	print("距离矩阵为：",diffMat)
	sqDiffMat = diffMat**2
	#sum()所有元素相加，sum(0)列相加，sum(1)行相加
	sqDistances = sqDiffMat.sum(axis=1)
	#开方，计算出距离
	distances = sqDistances**0.5
	#里面按照到测试点的距离的升序存放了数据点的index
	sortedDistIndices = distances.argsort()
	#定一个记录类别次数的字典
	classCount = {}
	for i in range(k):
		#取出前k个元素的类别
		voteIlabel = labels[sortedDistIndices[i]]
		#dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
		#计算类别次数
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
	#python3中用items()替换python2中的iteritems()
	#key=operator.itemgetter(1)根据字典的值进行排序
	#key=operator.itemgetter(0)根据字典的键进行排序
	#reverse降序排序字典
	sortedClassCount = sorted(classCount.items(),key=lambda x:x[1],reverse=True)
	#返回次数最多的类别,即所要分类的类别
	return sortedClassCount[0][0]

if __name__ == '__main__':
	#创建数据集
	group, labels = createDataSet()
	#测试集
	test = [101,20]
	#kNN分类
	test_class = classify0(test, group, labels, 3)
	#打印分类结果
	print(test_class)
