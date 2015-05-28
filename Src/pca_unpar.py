#!/usr/bin/env python
## -*- coding: utf-8 -*-

import sys, getopt
import numpy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.mlab import PCA as mlabPCA
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import pylab as pl

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def main(argv):
    #conf = SparkConf().setAppName('pca').setMaster('spark://albhed:7077')
    #sc = SparkContext(conf=conf)
    #inputfile = getArguments(argv)
    #Uncomment following line to enable argument
    #dict = unpickle(inputfile)
    
    global num_batch
    num_batch = 2
    dict = {};
    for x in range(num_batch):
        if len(dict) == 0:
            dict = unpickle('data_batch_' + str(x+1))
        else:
            tmp_dict = unpickle('data_batch_' + str(x+1))
            dict['data'] = numpy.vstack([dict['data'], tmp_dict['data']])
            dict['labels'].extend(tmp_dict['labels'])
    
    dt = unpickle('batches.meta')
    print dt
    #Initialize matrix
    size = len(dict['data'])
    featureSize = len(dict['data'][0])/3
    grayMatrix = numpy.zeros((size, featureSize))
    normalRGB = numpy.zeros((size, featureSize,3))
    
    #Compute intensity and normalize
    items = list(dict.items())
    dataItem = items[0]
    labels = numpy.array(dict['labels'])
    dataMatrix = numpy.array(dataItem[1])
    R = dataMatrix[:,:1024]
    G = dataMatrix[:,1024:2048]
    B = dataMatrix[:,2048:]
    #Construct original picture
    normalRGB = numpy.rollaxis(numpy.asarray([R,G,B]), 0,3)
    #Construct intensity Array
    grayMatrix = (R*0.2989+G*0.5870+B*0.1140)
    #Normalize
    grayMatrix -= grayMatrix.mean(axis=1)[:, None]
    
    #Apply PCA
    train(grayMatrix, featureSize, labels)

#Compute covariance matrix
#M = numpy.dot(grayMatrix,grayMatrix.T)
#e,EV = numpy.linalg.eigh(M)
# normal = normalRGB[0].reshape((32,32,3))
# plt.imshow(normal) #load
# plt.show()  # show the window
# grey = grayMatrix[0].reshape((32,32))
# plot(grey)


def plot(data):
    plt.imshow(data, cmap = cm.Greys_r)
    plt.show()

def getArguments(argv):
    if len(sys.argv) != 2:
        print 'Please pass training file'
        sys.exit()
    return str(sys.argv[1])

def train(matrix, featureSize, labels):
    #divide into train data and test data
    num_split = (num_batch - 1) * 10000
    matrix_test = matrix[num_split: (num_split + 10000),:]
    labels_test = labels[num_split: (num_split + 10000)]
    matrix = matrix[:num_split,:]
    labels = labels[:num_split]

    #Compute cov matrix
    cov_mat = numpy.cov(matrix.T)
    eig_val_cov, eig_vec_cov = numpy.linalg.eig(cov_mat)# Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(numpy.abs(eig_val_cov[i]), eig_vec_cov[:,i])]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort()
    eig_pairs.reverse()

    matrix_w = eig_pairs[0][1].reshape(featureSize,1)
    for i in range(10):
    	matrix_w = numpy.hstack((matrix_w, eig_pairs[i+1][1].reshape(featureSize,1)))
	# print matrix_w.shape
	# print matrix.dot(matrix_w)
	# print matrix_test.dot(matrix_w)

	# calculate projection
	transformed = matrix.dot(matrix_w)
	tf = transformed.T

	#recData = transformed.dot(matrix_w.T) + matrix.mean(axis=1)[:, None]
	#plot(recData[0].reshape((32,32)))

	#Start to train SVM
	#gamma = 1/sigma^2
	OVRC = OneVsRestClassifier(SVC(kernel = "rbf", gamma = 0.000001)).fit(transformed, labels)

	# Print Scatter plot
	# color = [str((item+1) * 24./255.) for item in labels]
	# pl.scatter(tf[0],tf[1],c = color )
	# pl.show()

	# Print the shape of classifier
	# h = 10
	# x_min, x_max = transformed[:, 0].min() - 1, transformed[:, 0].max() + 1
	# y_min, y_max = transformed[:, 1].min() - 1, transformed[:, 1].max() + 1
	# xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h), numpy.arange(y_min, y_max, h))
	# YY = OVRC.predict(numpy.c_[xx.ravel(), yy.ravel()])
	# print YY
	# YY = YY.reshape(xx.shape)
	# pl.contourf(xx, yy, YY, cmap=plt.cm.Paired, alpha=0.8)
	# pl.show()

	#Output training accuracy
	Y = OVRC.predict(matrix.dot(matrix_w))
	print Y
	correct = 0.0
	for x in range(len(Y)):
	    if labels[x] == Y[x]:
	        correct = correct +1
	print correct/len(Y)

	#Output test accuracy
	Z = OVRC.predict(matrix_test.dot(matrix_w))
	print Z
	correct = 0.0
	for x in range(len(Z)):
	    if labels_test[x] == Z[x]:
	        correct = correct +1
	print correct/len(Z)


#Main entry
if __name__ == "__main__":
    main(sys.argv)