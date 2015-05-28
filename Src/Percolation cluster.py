import sys
import numpy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
import math


class cluster(object):
	size = 10000 #size of sample
	Empty=-size*size-1
	ptr=numpy.ones(size)*Empty #array of pointers
	radius=3 #radius of neighbor
	

	def neighbor(data,size,num_features):
		ed=numpy.zeros([size,size]) #distance among samples
		radius=3 #radius of neighbor
		numneighbor=numpy.zeros(size) #number of neighbors of each sample
		indexneighbor=numpy.zeros([size,size]) #index of neighbors of each sample
		for i in range(0,size):
			for j in range(0,size):
				ed[i,j]=numpy.sqrt(numpy.dot((data[i,:]-data[j,:]).T,(data[i,:]-data[j,:])))
					if ed[i,j]<=radius and i!=j:
						indexneighbor[i,numneighbor[i] ]=j
						numneighbor[i] +=1
		return numneighbor,indexneighbor
	
	def cluster(data,size,num_features):
		numneighbor,indexneighbor=neighbor(data,size,num_features)
		print numneighbor
		order=numpy.zeros(size) 
		for i in range(0,size):
			r1=s1=i #sample i use itself as its root
			ptr[s1]=-1 #Initially, the cluster only contains i-th sample
			for j in range(0,int(numneighbor[i] )):
				s2=indexneighbor[i] [j] 
	
				if ptr[s2] is not Empty and math.isnan(ptr[s2]) is not True:
					r2=findroot(s2)
					if r2 is None:
						continue
					if r2 is not None:
						if r2!=r1:
							if ptr[r1]>ptr[r2]:
								ptr[r2]+=ptr[r1]
								ptr[r1]=r2
								r1=r2
							else:
								ptr[r1]+=ptr[r2]
								ptr[r2]=r1

		numofcluster=0	
		for i in range(0,size):
			if ptr[i] <0 and ptr[i] !=Empty:
				print ptr[i] 
				numofcluster+=1
		#print numofcluster
		return numofcluster

	def findroot(i):
		if math.isnan(ptr[i] ) is not True: 	
			if ptr[i] <0:
				return i
			else :
				ptr[i] =findroot(ptr[i] )

