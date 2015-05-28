 
def getArguments(argv):
 	if len(sys.argv) != 2:
 		print 'Please pass training file'
 		sys.exit()
 	return str(sys.argv[1])

def train(sc,matrix, featureSize, labels):
	matrix=matrix.T
	RDD1=sc.parallelize(matrix) #set up the graMatrix on the spark
	normalized_data=RDD1.map(normalized) #normalized the original matrix
	#normalized_data.collect()
	covData=normalized_data.map(covariance)
	#normalized_matrix=normalized(matrix)
	#cor=covariance(normalized_matrix)
	#print cor.shape
	c=covData.collect()
	covData.map(svd)
	#cc=numpy.asarray(c)
 	eig_val_cov, eig_vec_cov = numpy.linalg.eig(c)# Make a list of (eigenvalue, eigenvector) tuples
 	
	



def train1(sc,matrix, featureSize, labels):
 	cov_mat = numpy.cov(matrix.T)
 	print cov_mat.shape
 	eig_val_cov, eig_vec_cov = numpy.linalg.eig(cov_mat)# Make a list of (eigenvalue, eigenvector) tuples
 	eg, vec = svd(matrix)
 	ep = [(numpy.abs(eg[i]), vec[:,i]) for i in range(len(eg))]
 	ep.sort()
 	ep.reverse()

 	print ep
 	eig_pairs = [(numpy.abs(eig_val_cov[i]), eig_vec_cov[:,i]) for i in range(len(eig_val_cov))]
 
 	# Sort the (eigenvalue, eigenvector) tuples from high to low
 	eig_pairs.sort()
 	eig_pairs.reverse()

 	print eig_pairs
 	#recData = transformed.dot(matrix_w.T) + matrix.mean(axis=1)[:, None]
 	#plot(recData[0].reshape((32,32)))
 			
 #Main entry
if __name__ == "__main__":
     main(sys.argv)