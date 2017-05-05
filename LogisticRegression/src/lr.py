from numpy import *
import random
import matplotlib.pyplot as plt

def sigmoid(x):
	return 1 / (1 + exp(-x))

def LRtrain(train_x, train_y, opts):
	sample_num, feature_num = shape(train_x)
	alpha = opts['alpha']
	maxIter = opts['maxIter']
	weights = ones((feature_num, 1))

	for i in range(maxIter):
		if opts['optimizeType'] == 'gradDescent':
			alpha = 2.0 / (1 + i) + 0.01
			sig_res = sigmoid(train_x * weights)
			error = train_y - sig_res
			weights = weights + alpha * train_x.transpose() * error
		elif opts['optimizeType'] == 'stocGradDescent':
			for j in range(sample_num):
				sig_res = sigmoid(train_x[j, :] * weights)
				error = train_y[j, 0] - sig_res
				weights = weights + alpha * train_x[j, :].transpose() * error
		elif opts['optimizeType'] == 'smoothStocGradDescent':
			dataIndex = range(sample_num)
			for j in range(sample_num):
				# select one sample to modify theta randomly
				alpha = 4.0 / (1 + i + j) + 0.01
				randIndex = int(random.uniform(0, len(dataIndex)))
				sig_res = sigmoid(train_x[dataIndex[randIndex], :] * weights)
				error = train_y[dataIndex[randIndex], 0] - sig_res
				weights = weights + alpha * train_x[dataIndex[randIndex]].transpose() * error
				del(dataIndex[randIndex])
		else:
			raise NameError('No support function.')

	return weights

def testLR(weights, test_x, test_y):
	sample_num, feature_num = shape(test_x)
	accTotal = 0
	for i in xrange(sample_num):
		predict = sigmoid(test_x[i, :] * weights)[0, 0] > 0.5
		if predict == bool(test_y[i, 0]):
			accTotal += 1
	accuracy = float(accTotal / sample_num)
	return accuracy

def showLR(weights, train_x, train_y):  
    # notice: train_x and train_y is mat datatype  
    numSamples, numFeatures = shape(train_x)  
    if numFeatures != 3:  
        print "Sorry! I can not draw because the dimension of your data is not 2!"  
        return 1  
  
    # draw all samples  
    for i in xrange(numSamples):  
        if int(train_y[i, 0]) == 0:  
            plt.plot(train_x[i, 1], train_x[i, 2], 'or')  
        elif int(train_y[i, 0]) == 1:  
            plt.plot(train_x[i, 1], train_x[i, 2], 'ob')  
  
    # draw the classify line  
    min_x = min(train_x[:, 1])[0, 0]  
    max_x = max(train_x[:, 1])[0, 0]  
    weights = weights.getA()  # convert mat to array  
    y_min_x = float(-weights[0] - weights[1] * min_x) / weights[2]  
    y_max_x = float(-weights[0] - weights[1] * max_x) / weights[2]  
    plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')  
    plt.xlabel('X1'); plt.ylabel('X2')  
    plt.show()

def loadData():
	train_x = []
	train_y = []
	inputfile = open('../data/test.txt')
	for line in inputfile.readlines():
		lineArr = line.strip().split(' ')
		train_x.append([1.0, float(lineArr[0]), float(lineArr[1])])
		train_y.append(int(lineArr[2]))
	inputfile.close()
	return mat(train_x), mat(train_y).transpose()

def main():
	# 1. load data
	print "1. load data..."
	train_x, train_y = loadData()
	test_x = train_x
	test_y = train_y

	# 2. training
	print "2. training..."
	opts = {'alpha':0.01, 'maxIter':100, 'optimizeType':'smoothStocGradDescent'}
	weights = LRtrain(train_x, train_y, opts)
	print weights

	# 3. testing
	print "3. testing..."
	accuracy = testLR(weights, test_x, test_y)
	print accuracy

	# 4. show the result
	showLR(weights, train_x, train_y)



if __name__ == '__main__' :
	main()