#!/usr/bin/env python
# -*- coding: utf-8 -*-
#from __future__ import absolute_import, division, print_function
import scipy as sp
import evoMPS.tdvp_uniform as utdvp
import matplotlib.pyplot as plt
import math
import scipy.special as spe
'''
	This is simulation for calculating ground state for transeverse Ising model
'''

#Set up global variables for simulation
#N = 5								#number of site
bond_dim = 16 						#maximum bond dimension
h = 0.01 								#Field factor before quench

tol_im = 1E-6                		#Ground state tolerance (norm of projected evolution vector)

step = 0.08                   		#Imaginary time step size

auto_truncate = False          		#Whether to reduce the bond-dimension if any Schmidt coefficients fall below a tolerance.
zero_tol = 1E-20            		#Zero-tolerance for the Schmidt coefficients squared (right canonical form)

plot_results = True

sanity_checks = False         #Whether to perform additional (verbose) sanity checks

#Pauli matrices
Sx = sp.array([[0., 1.],
				 [1., 0.]])
Sy = 1.j * sp.array([[0., -1.],
					   [1., 0.]])
Sz = sp.array([[1., 0.],
				 [0., -1.]])

#define Hamiltonian
'''
A translationally invariant nearest-neighbour Hamiltonian is a 4-dimensional arrays, one for
each pair of sites.
For each term, the indices 0 and 1 are the 'bra' indices for the first and
second sites and the indices 2 and 3 are the 'ket' indices:

  ham[n][s,t,u,v] = <st|h|uv> (for sites n and n+1)
'''
def get_hamiltonian(h):
	ham = -(sp.kron(Sx, Sx) + h * sp.kron(Sz, sp.eye(2))).reshape(2, 2, 2, 2)
	return ham

'''
create an instance of the uniform evoMPS class.
'''
s = utdvp.EvoMPS_TDVP_Uniform(bond_dim,2,get_hamiltonian(h))
s.zero_tol = zero_tol
s.sanity_checks = sanity_checks


"""
The following loads a ground state from a file.
The ground state will be saved automatically when it is declared found.
"""
grnd_fname = "t_ising_uni_D%d_h%g_s%g_dtau%g_ground.npy" % (bond_dim, h, tol_im, step)


if __name__ == '__main__':
	
	print "Bond dimensions: " + str(s.D)
	print
	col_heads = ["Step", "t", "<H>", "eta"] #These last three are for testing the midpoint method.
	print "\t".join(col_heads)
	print
	t = 0
	T = []
	H = []
	
	i = 0
	eta = 1
	while True:
		T.append(t)	#Time vector

		s.update(auto_truncate=auto_truncate)

		H.append(s.h_expect.real)

		row = [str(i)]
		row.append(str(t))
		row.append("%.15g" % H[-1])
		if len(H) > 1:
			dH = H[-1] - H[-2]
		else:
			dH = 0

		


		s.take_step(step)
		t += 1.j * step
		eta = s.eta.real.sum()        
		row.append("%.6g" % eta)
		
		if math.fmod(i,10) == 0:
			print "\t".join(row)

		i += 1
		if eta < tol_im:
			s.save_state(grnd_fname)
			print "Ground state saved"
			break
	#print analytical solution
	lam = 1. / h
	print "Exact energy = ", -h * 2 / sp.pi * (1 + lam) * spe.ellipe((4 * lam / (1 + lam)**2))

	'''
		Plot the results
	'''
	tau = sp.array(T).imag
	fig1 = plt.figure(1)
	#fig2 = plt.figure(2)
	H_tau = fig1.add_subplot(111)
	H_tau.set_xlabel('tau')
	H_tau.set_ylabel('H')
	H_tau.set_title('Imaginary time evolution: Energy')
	#M_tau = fig2.add_subplot(111)
	#M_tau.set_xlabel('tau')
	#M_tau.set_ylabel('M')
	#M_tau.set_title('Imaginary time evolution: Magnetization')

	H_tau.plot(tau, H)
	#M_tau.plot(tau, M)
	plt.show()