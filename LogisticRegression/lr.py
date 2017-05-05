import numpy as np
import matplotlib.pyplot as plt
import time

def loadData():
	train_x = []
	train_y = []
	inputfile = open('../data/test1.txt')
	for line in inputfile.readlines():
		lineArr = line.strip().split('\t')
		train_x.append()

def main():
	# 1. load data
	print "1. load data..."
	train_x, train_y = loadData()
	test_x = train_x
	test_y = train_y

	# 2. training
	print "2. training..."
	opts = {'alpha':0.5, 'maxIter':100}
	weights = LRtrain(train_x, train_y)

	# 3. testing
	print "3. testing..."
	accuracy = testLR(weights, test_x, test_y)

	# 4. show the result
	showLR(weights, train_x, train_y)



if __name__ == '__main__' :
	main()