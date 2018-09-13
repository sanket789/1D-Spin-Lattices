# -*- coding: utf-8 -*-
#from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import math

'''
	Reference: arXiv:1805.02629
'''
def ent(n):
	try:
		S = n*math.log(n)
	except ValueError as ve:
		S = 0.
	return S

isAC = True

L = 2
J = 0.5
a = 1.0
F = 0.5
w = 0.21
tf = 10.
dt = 1.

Ft = a*F*np.arange(-L+0.5,L+1-0.5,1)

HH0 = np.diag(-J*np.ones(2*L-1),1) + np.diag(-J*np.ones(2*L-1),-1) 
evals0 , evec0 = np.linalg.eigh(HH0)

#psi0 = evec0[:,0]
'''
psi0 = np.zeros((2*L,1))
for n in range(0,2*L,2):
	psi0[n,0] = 1.0/np.sqrt(L)
'''
HH = HH0 + np.diag(Ft)
eps , DD = np.linalg.eigh(HH)

'''
l = range(0,2*L,2)
CC_0 =  np.zeros((2*L,2*L))##
for i in l:
	CC_0[i,i] = 1.
print CC_0
'''
CC_0 =  np.zeros((2*L,2*L))##
for i in range(0,L):
	CC_0[i,i] = 1.

UU = np.zeros((2*L,2*L)) 
N = int(tf/dt)
T = np.zeros(N)
S = np.zeros(N)
#print np.allclose(DD,DD.T)
CC_prev = CC_0.copy()
ttt = []
for i in range(N):
	T[i] = i*dt
	if isAC:
		Ft = a*F*math.cos(w*T[i])*np.arange(-L+0.5,L+1-0.5,1)
		HH = HH0 + np.diag(Ft)
		eps , DD = np.linalg.eigh(HH)
	if math.fmod(i,N/100.) == 0:
		print 100*i/N 
	
	
	EE = np.diag(np.exp(-1j*eps*dt))	#exp(-i_epsk_t)
	UU = np.dot(np.conj(DD),np.dot(EE,DD.T))	#derived
	#UU = np.dot(np.conj(DD),np.dot(EE,DD))		#from paper
	CC_t = np.dot(np.conj(UU).T, np.dot(CC_prev, UU)) 
	NN = np.linalg.eigvals(CC_t[0:L,0:L]).real

	UU_eq = np.dot(np.conj(DD),DD.T)
	print UU_eq
	CC_eq = np.dot(np.conj(UU).T, np.dot(CC_prev, UU)) 
	

	CC_prev = CC_t.copy()
	S[i] = sum([-ent((1-n)) - ent(n) for n in NN])
	ttt.append(CC_eq)
plt.plot(T,S)
plt.show()
for k in range(N):
	print ttt[k]
	#print np.allclose(ttt[0],ttt[k])
#np.save("ent_F%g_w%g_.npy"%(F,w),S)
#np.save("Time.npy",T)