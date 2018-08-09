#!/usr/bin/env python
# -*- coding: utf-8 -*-
#from __future__ import absolute_import, division, print_function
import scipy as sp
import evoMPS.tdvp_uniform as utdvp
import matplotlib.pyplot as plt
import math
import t_ising_uniform_ground as utg
import evoMPS.matmul as matmul
'''
	This code calculates the return probability for uniform transeverse ising model
'''


bond_dim = utg.bond_dim 					#maximum bond dimension
h = utg.h 								#Field factor before quench

tol_im = utg.tol_im                		#Ground state tolerance (norm of projected evolution vector)
step = utg.step                  		#Imaginary time step size

auto_truncate = False          			#Whether to reduce the bond-dimension if any Schmidt coefficients fall below a tolerance.
zero_tol = utg.zero_tol             		#Zero-tolerance for the Schmidt coefficients squared (right canonical form)

plot_results = True

sanity_checks = False         			#Whether to perform additional (verbose) sanity checks

h_quench = h							# after quench anisotropy parameter
tf = 10									#simulation final time
realstep = 0.0001 							#time step for real evolution
num_step = int(tf/realstep)

#Pauli matrices
Sx = utg.Sx
Sy = utg.Sy
Sz = utg.Sz

ham_quench = utg.get_hamiltonian(h_quench)

grnd_fname = "t_ising_uni_D%d_h%g_s%g_dtau%g_ground.npy" % (bond_dim, h, tol_im, step)
a_file = open(grnd_fname, 'rb')
utg.s.load_state(a_file)
a_file.close
grnd = sp.asarray(utg.s.A).copy()	#Ground state
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

def contract_uni(A,B):
	M = sp.dot(matmul.H(A[0][0]),B[0][0]) + sp.dot(matmul.H(A[0][1]),B[0][1])
	tr = sp.trace(matmul.H(A[0][0]))*sp.trace(B[0][0]) + sp.trace(matmul.H(A[0][1]))*sp.trace(B[0][1])
	return tr.real

print "h_initial = ",h,"\t","h_quench = ",h_quench
print "Ground state energy before quench = " , utg.s.h_expect.real ,
print "norm of ground state = ",contract_uni(grnd,grnd)
print "Maximum bond-dimension = " , utg.s.D
T = []	#time vector
P = []	#Return probability vector
H = [] 	#Eergy vector
rate = []
t = 0.
eta = 1.
utg.s.ham = ham_quench
print "Starting real time evolution"
print
col_heads = ["step","t","<H>","prob"] #These last three are for testing the midpoint method.
print "\t".join(col_heads)
print

for i in range(0,num_step):
	T.append(t)
	utg.s.update(auto_truncate=auto_truncate)

	H.append(utg.s.h_expect.real)

	if len(H) > 1 :
		dH = H[-1] - H[-2]
	else:
		dH = 0.
	#compute return probability
	prob = abs(contract_uni(grnd,sp.asarray(utg.s.A)))**2
	r = (1./1.)*math.log(1./prob)

	P.append(prob)
	rate.append(r)

	row = [str(i)]
	row.append(str(t))
	row.append("%.15g" % H[-1])
	row.append("%.15g" % prob)
	row.append("%.15g" % r)
	#dynamic expansion before Rk4 step
	dstep = realstep**(5./2.)
	utg.s.take_step(dstep * 1.j, dynexp=True, dD_max=16, sv_tol=1E-5)
	err = utg.s.etaBB.real
	utg.s.update(auto_truncate=False)

	utg.s.take_step_RK4((realstep - dstep) * 1.j)
	t = t + realstep

	eta = utg.s.eta.real 
	if math.fmod(i,10) == 0:
			print "\t".join(row)
print "real time evolution for" , tf, "seconds completed !"


'''
	Plot the results
'''
fig1 = plt.figure(1)
H_t = fig1.add_subplot(111)
H_t.set_xlabel('time in seconds')
H_t.set_ylabel('Energy')
H_t.plot(T,H)

fig2 = plt.figure(2)
P_t = fig2.add_subplot(111)
P_t.set_xlabel('time in seconds')
P_t.set_ylabel('probability')
P_t.plot(T,P)

fig3 = plt.figure(3)
R_t = fig3.add_subplot(111)
R_t.set_xlabel('time in seconds')
R_t.set_ylabel('Return rate')
R_t.plot(T,rate)

plt.show()