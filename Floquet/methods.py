import numpy as np
import math
def getHam_Efield_OBC(L,J,A,w,t,sanity_checks=False):

	Ft = A*math.cos(w*t)*np.arange(-L+0.5,L+1-0.5,1)
	HH0 = np.diag(-J*np.ones(2*L-1),1) + np.diag(-J*np.ones(2*L-1),-1)
	ham = HH0 + np.diag(Ft)
	if sanity_checks == True:
		if not np.allclose(np.conj(ham.T),ham):
			log.warning("Sanity Fail: Hamiltonian is not Hermitian")
	return ham

def getHam_Bfield_PBC(L,J,A,w,t,mu,a=1.0,sanity_checks=False):
	try:
		phi = 1j*a*A*np.sin(w*t)/w
	except ZeroDivisionError as ze:
		phi = 1j*a*A
		
	HH_hop = np.diag(-J*np.exp(-phi)*np.ones(2*L-1),1) + np.diag(-J*np.exp(phi)*np.ones(2*L-1),-1)
	ham = HH_hop + np.diag(mu*np.ones(2*L))
	ham[0,-1] = -J*np.exp(phi)
	ham[-1,0] = -J*np.exp(-phi) 

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

if __name__=='__main__':

	print (getHam_Bfield_PBC(2,0.1,0.5,0,2,1,a=1.0,sanity_checks=False))
