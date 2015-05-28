from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from pyspark import SparkContext
import cPickle
import sys
import numpy
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
import pyspark.rdd
num_features=100
 
def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict
 
def main(argv):
 	#conf = SparkConf().setAppName('pca').setMaster('spark://albhed:7077')
 	#sc = SparkContext(conf=conf)
 	sc=SparkContext("local","Boost")
 	#inputfile = getArguments(argv)
 	#Uncomment following line to enable argument
 	#dict = unpickle(inputfile)
 	dict = unpickle('/Users/nerohu/SkyDrive/00.JHU/CS_Learning/JHU_Courses/CS615_Bigdata/cifar-10-batches-py/data_batch_1')
 	dt = unpickle('/Users/nerohu/SkyDrive/00.JHU/CS_Learning/JHU_Courses/CS615_Bigdata/cifar-10-batches-py/batches.meta')
 	print dt
 	#Initialize matrix
 	size = len(dict['data'])
 	featureSize = len(dict['data'][0])/3
 	grayMatrix = numpy.zeros((size, featureSize))
 
 	#Compute intensity and normalize
 	items = list(dict.items())
 	dataItem = items[0]
 	labels = numpy.array(dict['labels'])
 	#print labels
 	dataMatrix = numpy.array(dataItem[1])
 	R = dataMatrix[:,:1024]
 	G = dataMatrix[:,1024:2048]
 	B = dataMatrix[:,2048:]
 	#Construct original picture
 	normalRGB = numpy.rollaxis(numpy.asarray([R,G,B]), 0,3)
 	#Construct intensity Array
 	grayMatrix = (R*0.2989+G*0.5870+B*0.1140)
 	#Normalize
 	#grayMatrix -= grayMatrix.mean(axis=1)[:, None]
 	RDD1=sc.parallelize(grayMatrix)
 	#meanVals=RDD1.map(numpy.mean).collect() #figure out mean
 	normalized_data=RDD1.map(normalized)
 	covData=normalized_data.map(covariance)
 	c=covData.collect()
 	print c.shape
 	e,EV=svd(c)
 	
 	#e,EV=eigsh(c,num_features,which='LM')
 	#evl,evc=svd(c)
 	#evc=sc.parallelize(c,3).map(svdevc).collect()

 	#print size(covData)
 	#print normalized_data
 	#meanVals=numpy.mean(grayMatrix)
 	#normalized_data=(grayMatrix-meanVals)/numpy.std(grayMatrix) #calculate normalized value
	#print normalized_data
	#covData=numpy.cov(normalized_data,rowvar=0) 
	#e,EV = eigsh(covData,num_features,which='LM')
 	#Apply PCA
 	#train(grayMatrix, featureSize, labels)
     
 	#Compute covariance matrix
 	#M = numpy.dot(grayMatrix,grayMatrix.T)
 	#e,EV = numpy.linalg.eigh(M)
 	#normal = normalRGB[0].reshape((32,32,3))
 	#plt.imshow(normal) #load
 	#plt.show()  # show the window
 	#grey = grayMatrix[0].reshape((32,32))
 	#plot(grey)
def covariance(data):
	cov=numpy.cov(data,rowvar=0)
	return cov

def eigenvalue(data):
	e,EV = eigsh(data,num_features,which='LM')
	return e



def eigenvector(data):
	e,EV=eigsh(data,num_features,which='LM')
	return EV

def normalized(data):
	meanVals=numpy.mean(data)
	std=numpy.std(data)
	normal=(data-meanVals)/std
	return normal

def svd(data):
	TOL=1e-8
	data=numpy.asarray(data)
	n = data.size
	print n
	#print n[1]
	maxiteration=60
	U=data
	print U.shape
	V=numpy.eye(n)
	singvals=numpy.zeros(n)
	converge=TOL+1
	countiteration=1
	while converge > TOL and countiteration<=maxiteration:
		countiteration=countiteration+1
		converge=0
		alpha=0
		beta=0
		gamma=0
		t=0
		for j in range(1,n):
			for i in range(0,j):
				for k in range(0,n):	
					alpha = alpha+U[k,i]*U[k,i]
					beta=beta+U[k,j]*U[k,j]
					gamma=gamma+U[k,i]*U[k,j]	
					converge=max(converge,abs(gamma)/numpy.sqrt(alpha*beta))
					#{alpha gamma;gamma beta}
					zeta=(beta-alpha)/(2*gamma)
				if zeta>0:
					t = 1/(numpy.abs(zeta)+numpy.sqrt(1+zeta*zeta))
				if zeta<0:
					t= -1/(numpy.abs(zeta)+numpy.sqrt(1+zeta*zeta))
					c = 1/numpy.sqrt(1+t^2)
					s = c*t

					#update columns i and j of U
					t=U[:,i]
					U[:,i]=c*t-s*U[:,j]
					U[:,j]=s*t-c*U[:,j]

					#update matrix V of eight sigunlar vectors
					t=V[:i]
					V[:,i]=c*t-s*V[:,j]
					V[:,j]=s*t+c*V[:,j]
					
		for j in range(0,n):
			singvals[j]=numpy.linalg.norm(U)
			U[:,j]=U[:,j]/singvals[j];
		return singvals,U

 
def train(matrix, featureSize, labels):
 	#Compute cov matrix
 	cov_mat = numpy.cov(matrix.T)
 	print 'shape' + str(cov_mat.shape)
 	eig_val_cov, eig_vec_cov = numpy.linalg.eig(cov_mat)# Make a list of (eigenvalue, eigenvector) tuples
 	eg, vec = svd(cov_mat)
 	ep = [(numpy.abs(eg[i]), vec[:,i]) for i in range(len(eg))]
 	ep.sort()
 	ep.reverse()

 	print ep
 	eig_pairs = [(numpy.abs(eig_val_cov[i]), eig_vec_cov[:,i]) for i in range(len(eig_val_cov))]
 
 	# Sort the (eigenvalue, eigenvector) tuples from high to low
 	eig_pairs.sort()
 	eig_pairs.reverse()

 	print eig_pairs
 
 	matrix_w = eig_pairs[0][1].reshape(featureSize,1)
 	for i in range(100):
 		matrix_w = numpy.hstack((matrix_w, eig_pairs[i+1][1].reshape(featureSize,1)))
 	print matrix_w.shape
 
 	transformed = matrix.dot(matrix_w)
 	print transformed.shape
 
 	#Start to train SVM
	# Z = OneVsRestClassifier(LinearSVC()).fit(transformed, labels).predict(transformed)
	Z = OneVsRestClassifier(SVC(kernel = "rbf")).fit(transformed, labels).predict(transformed)
 	print Z
 	correct = 0.0
 	for x in range(len(Z)):
 		if labels[x] == Z[x]:
 			correct = correct +1

 	print correct/len(Z)
 	#recData = transformed.dot(matrix_w.T) + matrix.mean(axis=1)[:, None]
 	#plot(recData[0].reshape((32,32)))
 			
 #Main entry
if __name__ == "__main__":
     main(sys.argv)