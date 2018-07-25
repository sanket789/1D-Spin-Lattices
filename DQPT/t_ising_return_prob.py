#!/usr/bin/env python
# -*- coding: utf-8 -*-
#from __future__ import absolute_import, division, print_function
import scipy as sp
import evoMPS.tdvp_gen as tdvp
import matplotlib.pyplot as plt
import math
import t_ising_ground as tg
import evoMPS.matmul as matmul
'''
	This code calculates the return probability for transeverse ising model
'''

N = tg.N								#number of site
bond_dim = tg.bond_dim 					#maximum bond dimension
h = tg.h 								#Field factor before quench

tol_im = tg.tol_im                		#Ground state tolerance (norm of projected evolution vector)
step = tg.step                  		#Imaginary time step size

auto_truncate = True          			#Whether to reduce the bond-dimension if any Schmidt coefficients fall below a tolerance.
zero_tol = tg.zero_tol             		#Zero-tolerance for the Schmidt coefficients squared (right canonical form)

plot_results = True

sanity_checks = False         			#Whether to perform additional (verbose) sanity checks

h_quench = 100							# after quench anisotropy parameter
tf = 10									#simulation final time
realstep = 0.1 							#time step for real evolution
num_step = int(tf/realstep)

#Pauli matrices
Sx = tg.Sx
Sy = tg.Sy
Sz = tg.Sz

ham_quench = tg.get_hamiltonian(N,h_quench)

grnd_fname = "t_ising_N%d_D%d_h%g_s%g_dtau%g_ground.npy" % (N, bond_dim, h, tol_im, step)
a_file = open(grnd_fname, 'rb')
tg.s.load_state(a_file)
a_file.close
grnd = tg.s.A.copy()	#Ground state
'''
	Function to multiply three matrices
'''
def mult(A,B,C):
	return sp.dot(A,sp.dot(B,C))
'''
	Function to evaluate contractions. Ref: Eq 93 from U. Schollowk
	Input:
		A : bra vector MPS
		B : ket vector MPS
	Output:
		< psi[A] | phi[B] >
'''
def contract(A,B):
	M0 = sp.dot(matmul.H(A[1][0]),B[1][0]) + sp.dot(matmul.H(A[1][1]),B[1][1])
	for n in range(2,N+1):
		Ml = A[n]
		Mr = B[n]
		out = mult(matmul.H(Ml[0]),M0,Mr[0]) + mult(matmul.H(Ml[1]),M0,Mr[1])
		M0 = out
	return sp.asscalar(M0)

print contract(grnd,grnd)
print tg.s.H_expect.real ,"Ground state energy before quench"
T = []	#time vector
R = []	#Return probability vector
H = [] 	#Eergy vector
t = 0.
eta = 1.
tg.s.ham = ham_quench
print "Starting real time evolution"
print
col_heads = ["t","<H>","prob"] #These last three are for testing the midpoint method.
print "\t".join(col_heads)
print

for i in range(0,num_step):
	T.append(t)
	tg.s.update(auto_truncate=auto_truncate)

	H.append(tg.s.H_expect.real)

	if len(H) > 1 :
		dH = H[-1] - H[-2]
	else:
		dH = 0.
	#compute return probability
	prob = abs(contract(grnd,tg.s.A))**2
	R.append(prob)
	row = []
	row.append(str(t))
	row.append("%.15g" % H[-1])
	row.append("%.15g" % prob)
	#dynamic expansion before Rk4 step
	dstep = realstep**(5./2.)
	tg.s.take_step(dstep * 1.j, dynexp=True, dD_max=4, sv_tol=1E-5)
	err = tg.s.etaBB.real.sum()
	tg.s.update(auto_truncate=False)

	tg.s.take_step_RK4((realstep - dstep) * 1.j)
	t = t + realstep

	eta = tg.s.eta.real.sum() 
	if math.fmod(i,10) == 0:
			print "\t".join(row)
print "real time evolution for" , 'tf', "seconds completed !"

'''
	Plot the results
'''
fig1 = plt.figure(1)
fig2 = plt.figure(2)

H_t = fig1.add_subplot(111)
P_t = fig2.add_subplot(111)
H_t.set_xlabel('time in seconds')
H_t.set_ylabel('Energy')
P_t.set_xlabel('time in seconds')
P_t.set_ylabel('probability')

H_t.plot(T,H)
P_t.plot(T,R)
plt.show()