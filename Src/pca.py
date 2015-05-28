#!/usr/bin/env python
## -*- coding: utf-8 -*-

import os
import sys, getopt
import numpy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.mlab import PCA as mlabPCA
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from PIL import Image
import pickle

from pyspark import SparkContext, SparkConf

from spylearn.blocked_math import count, cov, svd, svd_em
from spylearn.block_rdd import block_rdd

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def main(argv):
	conf = SparkConf().setAppName('pca').setMaster('local')
	global sc
	sc = SparkContext(conf=conf)
	#inputfile = getArguments(argv)
	#Uncomment following line to enable argument
	#dict = unpickle(inputfile)
	# dict = {};
	# for x in range(5):
	# 	if len(dict) == 0:
	# 		dict = unpickle('data_batch_' + str(x+1))
	# 	else:
	# 		tmp_dict = unpickle('data_batch_' + str(x+1))
	# 		dict['data'] = numpy.vstack([dict['data'], tmp_dict['data']])
	# 		dict['labels'] = numpy.vstack([dict['labels'], tmp_dict['labels']])
	# 	print len(dict['data'])

	dict = unpickle('data_batch_1')
	dt = unpickle('batches.meta')
	print dt
	
	i = Image.open("test.png")
	i = i.convert('RGB')
	pix=i.load()
	w=i.size[0]
	h=i.size[1]
	RT = numpy.zeros(w*h)
	BT = numpy.zeros(w*h)
 	GT = numpy.zeros(w*h)
	for i in range(h):
		for j in range(w):
			r,g,b = pix[j,i]
			RT[i*32 + j] = r
			GT[i*32 + j] = g
			BT[i*32 + j] = b

	grayMatrix = (RT*0.2989+GT*0.5870+BT*0.1140)
	#Normalize
	mean = grayMatrix.mean()
	grayMatrix -= grayMatrix.mean()

	#Initialize matrix
	size = len(dict['data'])
	featureSize = len(dict['data'][0])/3
	grayMatrix = numpy.zeros((size, featureSize))
	normalRGB = numpy.zeros((size, featureSize,3))

	#Compute intensity and normalize
	items = list(dict.items())
	dataItem = items[0]
	labels = numpy.array(dict['labels'])
	print labels
	dataMatrix = numpy.array(dataItem[1])
	R = dataMatrix[:,:1024]
	print R.shape
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
	normal = normalRGB[0].reshape((32,32,3))

	plt.imshow(normal) #load
	plt.show()  # show the window
	grey = grayMatrix[0].reshape((32,32))
	plot(grey)

def plot(data):	
	plt.imshow(data, cmap = cm.Greys_r)
	plt.show()

def getArguments(argv):
	if len(sys.argv) != 2:
		print 'Please pass training file'
		sys.exit()
	return str(sys.argv[1])

def train(matrix, featureSize, labels):

	diSmatrix = sc.parallelize(list(matrix), 10)

	#use spyleanr to parallelize SVD on RDD
	data = block_rdd(diSmatrix)
	u, s, v = svd(data, 100)

	print v.shape
	#Old unparallelized version
	# cov_mat = numpy.cov(matrix.T)
	# print cov_mat.shape
	# eig_val_cov, eig_vec_cov = numpy.linalg.eig(cov_mat)
	# # Make a list of (eigenvalue, eigenvector) tuples
	# eig_pairs = [(numpy.abs(eig_val_cov[i]), eig_vec_cov[:,i]) for i in range(len(eig_val_cov))]

	# # Sort the (eigenvalue, eigenvector) tuples from high to low
	# eig_pairs.sort()
	# eig_pairs.reverse()

	# matrix_w = eig_pairs[0][1].reshape(featureSize,1)
	# for i in range(200):
	#     matrix_w = numpy.hstack((matrix_w, eig_pairs[i+1][1].reshape(featureSize,1)))
	# print matrix_w.shape

	transformed = matrix.dot(v.T)
	print transformed.shape
	#Compute cov matrix
	# if os.path.isfile('svm.model'):
	#     print 'Loading Model file...'
	#     #Load models from file
	#     # with open('svm.model', 'rb') as file:
	#     #     Z = pickle.load(file)
	# else:
		#Start to train SVM
	Z = OneVsRestClassifier(SVC(kernel="rbf")).fit(transformed, labels)
	    # with open('svm.model', 'wb') as file:
	    #     pickle.dump(Z, file)

	Z = Z.predict(transformed)

	print Z[0]
	correct = 0.0
	for x in range(len(Z)):
		if labels[x] == Z[x]:
			correct = correct +1

	print correct/len(Z)

	print 'plot reconstructed data'
	recData = transformed.dot(v.T) + matrix.mean(axis=1)[:, None]
	plot(recData[0].reshape((32,32)))
			
#Main entry
if __name__ == "__main__":
    main(sys.argv)