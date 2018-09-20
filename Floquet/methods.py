import numpy as np
import math
def getHam_Efield_OBC(L,J,A,w,t,sanity_checks=False):

	Ft = A*math.cos(w*t)*np.arange(-L//2+0.5,L//2+1-0.5,1)
	HH0 = np.diag(-J*np.ones(L-1),1) + np.diag(-J*np.ones(L-1),-1)
	ham = HH0 + np.diag(Ft)
	if sanity_checks == True:
		if not np.allclose(np.conj(ham.T),ham):
			log.warning("Sanity Fail: Hamiltonian is not Hermitian")
	return ham

def getHam_Bfield_PBC(L,J,A,w,t,mu,a=1.0,sanity_checks=False):
	try:
		phi = 1j*a*A*getPhi(w,t,L,"B_FIELD")
	except ZeroDivisionError as ze:
		phi = 1j*a*A

	HH_hop = np.diag(-J*np.exp(-phi)*np.ones(L-1),1) + np.diag(-J*np.exp(phi)*np.ones(L-1),-1)
	ham = HH_hop + np.diag(mu*np.ones(L))
	ham[0,-1] = -J*np.exp(phi)
	ham[-1,0] = -J*np.exp(-phi) 

	if sanity_checks == True:
		if not np.allclose(np.conj(ham.T),ham):
			log.warning("Sanity Fail: Hamiltonian is not Hermitian")
	return ham

def getHam_flux_PBC(L,J,w,t,mu0,sigma,alpha,sanity_checks=False):

	phi = getPhi(w,t,L,"FLUX")
	mu_i = [mu0*np.cos(2*np.pi*sigma*index + alpha) for index in range(0,L)]	#site dependent quasi-periodic chemical potential
	HH_hop = np.diag(-J*np.exp(-1j*phi)*np.ones(L-1),1) + np.diag(-J*np.exp(1j*phi)*np.ones(L-1),-1)	#hopping terms
	ham = HH_hop + np.diag(mu_i)	#adding off-diagonal and diagonal part
	ham[0,-1] = -J*np.exp(1j*phi)	#PBC
	ham[-1,0] = -J*np.exp(-1j*phi) #PBC

	if sanity_checks == True:
		if not np.allclose(np.conj(ham.T),ham):
			log.warning("Sanity Fail: Hamiltonian is not Hermitian")
	return ham

def ent(n):
	#	n is float scalar
	try:
		S = n*math.log(n)
	except ValueError as ve:
		S = 0.
	return S


def vonNeumannEntropy(A,l0=int,l1=int):
	'''
		Calculate vonNeumann Entropy using formula: Eq17 from arXiv:1805.02629 
		Input:	Density matrix for full system. initial and final index of subsystem
		Output: Entropy and Eigenvalue ndarray for A[l0:l1,l0:l1] 
	'''
	NN = np.linalg.eigvals(A[l0:l1,l0:l1]).real
	S = sum([-ent((1-n)) - ent(n) for n in NN])
	return S , NN

def distance(A,B):
	'''
		Calculates distance between two density matrices defined as: d = |A-B|/sqrt(|A|**2 + |B|**2)
		norm is a Frobenius norm
		Input: A and B arrays
		Output: distance scalar
	'''
	#d = np.linalg.norm(A-B)/np.sqrt(np.linalg.norm(A)**2 + np.linalg.norm(B)**2)
	a = np.linalg.norm(A-B)
	b = np.sqrt(np.linalg.norm(A)**2 + np.linalg.norm(B)**2)
	d = (a/b) if b != 0 else 0
	
	return d


def getPhi(w,t,L,s):
	
	if s is "FLUX":
		phi = np.sin(w*t)/(L)	#phase
	elif s is "B_FIELD":
		phi = np.sin(w*t)/w
	else:
		phi = 1.0
	return phi


def getCurrent(CC,w,t,J,ss):
	L = np.shape(CC)[0]
	j = np.zeros(L)
	phi = getPhi(w,t,L,ss)
	j[0] = 1j*J*(np.exp(1j*phi)*CC[0,L] - np.exp(-1j*phi)*C[L,0])
	for i in range(1,L):
		j[i] = 1j*J*(np.exp(1j*phi)*CC[i,i-1] - np.exp(-1j*phi)*CC[n-1,n])

	return j

if __name__=='__main__':
	print(getPhi(3.14,1.,10,"FLUX"))