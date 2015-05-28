def svd(data):
	TOL=1e-8
	data=numpy.asarray(data)
	n=data.shape
	#print n[1]
	maxiteration=60
	U=data
	V=numpy.eye(n[0])
	singvals=numpy.zeros(n[0])
	print V
	converge=TOL+1
	countiteration=1
	while converge > TOL and countiteration<=maxiteration:
		countiteration=countiteration+1
		converge=0
		alpha=0
		beta=0
		gamma=0
		t=0
		for j in range(1,n[1]):
			for i in range(0,j):
				for k in range(0,n[1]):	
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

		for j in range(0,n[1]):
			singvals[j]=numpy.linalg.norm(U)
			U[:,j]=U[:,j]/singvals[j];
		return singvals,U
		#print singvals
		#print U