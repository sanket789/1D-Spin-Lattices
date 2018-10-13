import numpy as np
import matplotlib.pyplot as plt

L 	= 4
J 	= 1.
mu 	= 0.1
t 	= 0.
N = 2
nbar = np.zeros(N,dtype=float)
T = np.zeros(N)

CC 	= 	np.zeros((L,L))
for k in range(0,L,2):
	CC[k,k] = 1.0
print(np.sum(CC.diagonal()).real)
#print(CC)

HH = np.diag(-J*np.ones(L-1),-1) + np.diag(-J*np.ones(L-1),1) + np.diag(mu*np.ones(L))
HH[0,L-1] = -J
HH[L-1,0] = -J

eps , DD = np.linalg.eigh(HH)							#Find eigen values and eigenvevtors
EE = np.diag(np.exp((-1j*t)*eps))						#exp(-i_epsk_t)
UU = np.dot(np.conj(DD),np.dot(EE,DD.T))				#derived
print(np.dot(DD,np.conj(DD.T)))
#tmp = CC.copy()
for i in range(N):
	T[i] = i*t
	nbar[i] = np.sum(CC.diagonal()).real -L//2
	CC_next = (np.dot(np.conj(UU).T, np.dot(CC, UU)) )	#calculate CC for t+dt
	#print(np.allclose(CC,CC_next))
	#print(CC_next)
	#print(np.allclose(CC,tmp))
	#tmp = CC.copy()

	CC = CC_next.copy()

plt.plot(T,nbar)
plt.show()