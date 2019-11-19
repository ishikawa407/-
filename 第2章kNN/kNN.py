from numpy import *
import operator
import os


def datingClasstest():
    """
    测试算法
    """
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],
                                     datingLabels[numTestVecs:m], 3)
        print('the classifier came back with: %d, the real answer is: %d' \
              % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))


def classifyPerson():
    resultList = ["not at all", "in small doses", "in large doses"]
    percentTats = float(input(
        "percentage of time spent playing video games?"
    ))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year"))
    datingDataMat, datingLabels = file2matrix("datingTestSet2.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges,
                                 normMat, datingLabels, 3)
    print("You will probably like this person: ",
          resultList[classifierResult - 1])


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ["A", "A", "B", "B"]
    return group, labels


def classify0(inX, dataSet, labels, k):
    """
    1, 计算已知类别数据集中的每个点与当前点之间的距离；
    2  按照距离递增次序排列
    3  选取与当前点距离最小的k个点
    4  确定前k个点所在类别的出现频率
    5  返回前k点出现频率最高的类别作为当前点的预测分类
    :param inX:
    :param dataSet:
    :param labels:
    :param k:
    :return:
    """
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # np.tile
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    #print(classCount.items())
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1),
                              reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    with open(filename, 'r') as fr:
        arrayOLines = fr.readlines()
        numberOfLines = len(arrayOLines)
        returnMat = zeros((numberOfLines, 3))
        classLableVector = []

        for index, line in enumerate(arrayOLines):
            line = line.strip()
            listFromLine = line.split('\t')
            returnMat[index, :] = listFromLine[:3]
            classLableVector.append(int(listFromLine[-1]))
        return returnMat, classLableVector


def autoNorm(dataSet):
    minVals = dataSet.min(0)  # axis = 0
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals



"""
手写识别系统
"""


def img2vector(filename):
    returnVect =zeros((1, 1024))
    with open(filename, "r") as fr:
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwlabels = []
    trainingFileList = os.listdir("digits/trainingDigits")
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split(".")[0]
        classNumStr = int(fileStr.split('_')[0])
        hwlabels.append(classNumStr)
        trainingMat[i, :] = img2vector("digits/trainingDigits/%s" % fileNameStr)
    testFileList = os.listdir('digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split("_")[0])
        vectorUndetTest = img2vector('digits/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUndetTest,
                                     trainingMat, hwlabels, 3)
        print("the classifier came back with: %d, the real answer is: %d"\
              % (classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is : %f" % (errorCount/float(mTest)))