import numpy as np
import math as math
import matplotlib.pyplot as plt

N = 6	#number of sites
V = 0.
t = 1.


def getHam(N,t,V):
	'''
	Input:
		N = number of sites
		t = nearest neighbour interaction
		V = self interaction
	Output:
		N*N matrix 
	'''
	H = np.zeros((N,N))
	
	#zeroth and (N-1)th diagonal element
	H[0][0] = V
	H[N-1][N-1] = V
	#zeroth row
	H[0][1] = -t
	H[0][N-1] = -t
	#(N-1)th row
	H[N-1][0] = -t
	H[N-1][N-2] = -t

	for i in range(1,N-1):
		H[i][i] = V
		H[i][i+1] = -t
		H[i][i-1] = -t

	return H


H = getHam(N,t,V)
if np.allclose(H.T,H):
	print "Hermitian Hamiltonian !!"
else:
	print "Bad Hamiltonian!!!"
#eigvalue:	ndarray with ascending eigenvalue
#eigvec:	eigvec[:,i] is eigenvector corresponding to eigvalue[i]
eigvalue, eigvec = np.linalg.eigh(H)

index = range(N)

for k in range(N):
	plt.plot(index,abs(eigvec[:,k])**2)
	plt.pause(1.0)


plt.show()
print H
