#!/usr/bin/env python
# -*- coding: utf-8 -*-
#from __future__ import absolute_import, division, print_function
import scipy as sp
import evoMPS.tdvp_gen as tdvp
import matplotlib.pyplot as plt
import math
'''
	This is simulation for calculating ground state for transeverse Ising model
'''

#Set up global variables for simulation
N = 10								#number of site
bond_dim = 8 						#maximum bond dimension
h = 0.01 							#Field factor before quench

tol_im = 1E-10                		#Ground state tolerance (norm of projected evolution vector)

step = 0.08                   		#Imaginary time step size

auto_truncate = True          		#Whether to reduce the bond-dimension if any Schmidt coefficients fall below a tolerance.
zero_tol = 1E-12              		#Zero-tolerance for the Schmidt coefficients squared (right canonical form)

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
A nearest-neighbour Hamiltonian is a sequence of 4-dimensional arrays, one for
each pair of sites.
For each term, the indices 0 and 1 are the 'bra' indices for the first and
second sites and the indices 2 and 3 are the 'ket' indices:

  ham[n][s,t,u,v] = <st|h|uv> (for sites n and n+1)
'''
def get_hamiltonian(num_sites,h):
	ham = -(sp.kron(Sx, Sx) + h * sp.kron(Sz, sp.eye(2))).reshape(2, 2, 2, 2)
  	ham_end = ham + h*sp.kron(sp.eye(2), Sz).reshape(2, 2, 2, 2)
	return [None] + [ham] * (num_sites - 2) + [ham_end]

v_D = [bond_dim] * (N + 1)	#The bond dimension for each site is given as a vector, length N + 1.

v_q = [2] * (N + 1)			#The site Hilbert space dimension is also given as a vector, length N + 1.

'''
create an instance of the evoMPS class.
'''
s = tdvp.EvoMPS_TDVP_Generic(N, v_D, v_q, get_hamiltonian(N, h))
s.zero_tol = zero_tol
s.sanity_checks = sanity_checks


"""
The following loads a ground state from a file.
The ground state will be saved automatically when it is declared found.
"""
grnd_fname = "t_ising_N%d_D%d_h%g_s%g_dtau%g_ground.npy" % (N, bond_dim, h, tol_im, step)


if __name__ == '__main__':
	
	print "Bond dimensions: " + str(s.D)
	print
	col_heads = ["Step", "t", "<H>", "d<H>",
				 "sig_x_3", "sig_y_3", "sig_z_3",
				 "M_x", "eta"] #These last three are for testing the midpoint method.
	print "\t".join(col_heads)
	print
	t = 0
	T = []
	H = []
	M = []
	i = 0
	eta = 1
	while True:
		T.append(t)	#Time vector

		s.update(auto_truncate=auto_truncate)

		H.append(s.H_expect.real)

		row = [str(i)]
		row.append(str(t))
		row.append("%.15g" % H[-1])
		if len(H) > 1:
			dH = H[-1] - H[-2]
		else:
			dH = 0

		row.append("%.2e" % (dH.real))
		"""
		Compute expectation values!
		"""
		Sx_3 = s.expect_1s(Sx, 3) #Spin observables for site 3.
		Sy_3 = s.expect_1s(Sy, 3)
		Sz_3 = s.expect_1s(Sz, 3)
		row.append("%.3g" % Sx_3.real)
		row.append("%.3g" % Sy_3.real)
		row.append("%.3g" % Sz_3.real)

		m_n = [s.expect_1s(Sz, n).real for n in range(1, N + 1)] #Magnetization
		m = sp.sum(m_n)

		row.append("%.9g" % m)
		M.append(m)

		s.take_step(step, calc_Y_2s=True)
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
	if h == 1:	
		E = - 2 * abs(sp.sin(sp.pi * (2 * sp.arange(N) + 1) / (2 * (2 * N + 1)))).sum()
		print "Exact ground state energy = %.15g" % E

	'''
		Plot the results
	'''
	tau = sp.array(T).imag
	fig1 = plt.figure(1)
	fig2 = plt.figure(2)
	H_tau = fig1.add_subplot(111)
	H_tau.set_xlabel('tau')
	H_tau.set_ylabel('H')
	H_tau.set_title('Imaginary time evolution: Energy')
	M_tau = fig2.add_subplot(111)
	M_tau.set_xlabel('tau')
	M_tau.set_ylabel('M')
	M_tau.set_title('Imaginary time evolution: Magnetization')

	H_tau.plot(tau, H)
	M_tau.plot(tau, M)
	plt.show()