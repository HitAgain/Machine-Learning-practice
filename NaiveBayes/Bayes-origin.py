# -*- coding: UTF-8 -*-
import numpy as np
"""
以下代码采用最原始的Bayes分类  将文档转换为对应向量时  只用0，1表示字典词汇的出现与否  不计数具体个数
"""
"""
函数说明:创建实验样本
"""
def loadDataSet():  ## 生成样本标签 和数据集
	postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],				#切分的词条
				['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
				['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
				['stop', 'posting', 'stupid', 'worthless', 'garbage'],
				['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
				['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
	classVec = [0, 1, 0, 1, 0, 1]   																#类别标签向量，1代表侮辱性词汇，0代表不是
	return postingList, classVec																#返回实验样本切分的词条和类别标签向量

"""
函数说明:将切分的实验样本词条整理成不重复的词条列表，也就是词汇表

Parameters:
	dataSet - 整理的样本数据集
Returns:
	vocabSet - 返回不重复的词条列表，也就是词汇表
"""
def createVocabList(dataSet):
	vocabSet = set([])  					#创建一个空的不重复列表
	for document in dataSet:
		vocabSet = vocabSet | set(document) #取并集，去重
	return list(vocabSet)                   # 返回字典



"""
函数说明:根据vocabList词汇表，将inputSet向量化，向量的每个元素为1或0

Parameters:
	vocabList - createVocabList返回的列表
	inputSet - 切分的词条列表
Returns:
	returnVec - 文档向量,词集模型
"""
def setOfWords2Vec(vocabList, inputSet):
	returnVec = [0] * len(vocabList)									#创建一个其中所含元素都为0的向量
	for word in inputSet:												#遍历每个词条
		if word in vocabList:											#如果词条存在于词汇表中，则置1
			returnVec[vocabList.index(word)] = 1
		else: print("the word: %s is not in my Vocabulary!" % word)
	return returnVec													#返回文档向量，任何文档向量都是等长的 等于vacabulist的长度
"""
函数说明:朴素贝叶斯分类器训练函数
"""
def trainNB0(trainMatrix,trainCategory):
	numTrainDocs = len(trainMatrix)							#计算训练的文档数目
	numWords = len(trainMatrix[0])							#字典大小 总词汇的规模
	pAbusive = sum(trainCategory)/float(numTrainDocs)		# P(y=1)

	p1Num_nega = np.zeros(numWords)
	p1Num_posi = np.zeros(numWords)
	nega_count = 0.0
	posi_count = 0.0

	for i in range(numTrainDocs):
		if trainCategory[i] == 1:
			p1Num_nega += trainMatrix[i]
			#p1Denom += sum(trainMatrix[i])
			nega_count += 1.0
		else:
			p1Num_posi += trainMatrix[i]
			#p0Denom += sum(trainMatrix[i])
			posi_count += 1.0
	nega_1_vect = p1Num_nega/nega_count                              # 对任意i，P(xi=1|y=1)
	nega_0_vect = np.ones(numWords, dtype=np.float32) - nega_1_vect  # 对任意i，P(xi=0|y=1)
	posi_1_vect = p1Num_posi/posi_count                              # 对任意i，P(xi=1|y=0)
	posi_0_vect = np.ones(numWords,dtype=np.float32) - posi_1_vect   # 对任意i，P(xi=0|y=0)
	total_1 = (p1Num_nega+p1Num_posi)/float(numTrainDocs)            # 对任意i，P(x1 = 1)
	total_0 =  np.ones(numWords,dtype=np.float32) - total_1          # 对任意i，P(x1 = 0)
	return nega_1_vect, nega_0_vect, posi_1_vect, posi_0_vect, total_1, total_0, pAbusive

"""
函数说明:朴素贝叶斯分类器分类函数
Returns:
	0 - 属于非侮辱类
	1 - 属于侮辱类
"""
def classifyNB(vec2Classify,nega_1_vect, nega_0_vect, posi_1_vect, posi_0_vect, total_1, total_0, pClass1):
	p_text = 1.0
	for i in range(len(vec2Classify)):
		if vec2Classify[i] == 1:
			p_text *= total_1[i]
		else:
			p_text *= total_0[i]
	p_text_1 = 1.0
	for i in range(len(vec2Classify)):
		if vec2Classify[i] == 1:
			p_text_1 *= nega_1_vect[i]
		else:
			p_text_1 *= nega_0_vect[i]
	P_is_posi = (p_text_1 * pClass1)/p_text
	if P_is_posi > 0.5:
		return 1
	else:
		return 0



def testingNB():
	listOPosts,listClasses = loadDataSet()									#创建实验样本
	myVocabList = createVocabList(listOPosts)								#创建词汇表
	trainMat=[]
	for postinDoc in listOPosts:
		trainMat.append(setOfWords2Vec(myVocabList, postinDoc))				#将实验样本向量化
	nega_1_vect, nega_0_vect, posi_1_vect, posi_0_vect, total_1, total_0, pAbusive = trainNB0(np.array(trainMat),np.array(listClasses))		#训练朴素贝叶斯分类器
	testEntry = ['love', 'my', 'dalmation']									#测试样本1
	thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))				#测试样本向量化
	if classifyNB(thisDoc,nega_1_vect, nega_0_vect, posi_1_vect, posi_0_vect, total_1, total_0, pAbusive):
		print(testEntry,'属于侮辱类')										#执行分类并打印分类结果
	else:
		print(testEntry,'属于非侮辱类')										#执行分类并打印分类结果

	testEntry = ['stupid', 'garbage']										#测试样本2
	thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))				#测试样本向量化
	if classifyNB(thisDoc, nega_1_vect, nega_0_vect, posi_1_vect, posi_0_vect, total_1, total_0, pAbusive):
		print(testEntry, '属于侮辱类')										#执行分类并打印分类结果
	else:
		print(testEntry, '属于非侮辱类')										#执行分类并打印分类结果

if __name__ == '__main__':
	testingNB()
