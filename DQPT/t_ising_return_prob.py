#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os, datetime
import scipy as sp
import evoMPS.tdvp_gen as tdvp
import matplotlib.pyplot as plt
import math
import DQPT.t_ising_ground as tg
import evoMPS.matmul as matmul
import time




'''
	This code calculates the return probability for transeverse ising model with spin 1/2.
	This code uses generic evoMPS class for MPS formulation.
'''
##parameters for imaginary time evolution
N = 6									#number of site
bond_dim = 8							#maximum bond dimension

J = 0.0 								#J-paramter of ising Hamiltonian
h = 1.0	 								#Field factor before quench

tol_im = 1e-10	                		#Ground state tolerance (norm of projected evolution vector)
step_im = 0.0009765625	                  #Imaginary time step size

auto_truncate = False          			#Whether to reduce the bond-dimension if any Schmidt coefficients fall below a tolerance.
zero_tol = 1e-20	             		#Zero-tolerance for the Schmidt coefficients squared (right canonical form)


##parameters for real time evolution
J_quench = 1.0
h_quench = 0.0							# after quench anisotropy parameter
tf = 10.0								#simulation final time

#realstep = 0.0009765625			#time step for real evolution
realstep = 0.001953125	

num_step = int(tf/realstep)

plot_results = False
sanity_checks = False         			#Whether to perform additional (verbose) sanity checks
save_data = False 
#--------------------------------------------------------------------------------------------
print("Simulation for quench from J = %g ,h = %g to J = %g , h = %g with %g sites and %g bond dimension" % 
																		(J,h,J_quench,h_quench,N,bond_dim))

#Pauli matrices
Sx = sp.array([[0., 1.],
				 [1., 0.]])
Sy = 1.j * sp.array([[0., -1.],
					   [1., 0.]])
Sz = sp.array([[1., 0.],
				 [0., -1.]])

grnd_fname = "t_ising_N%d_D%d_J%g_h%g_s%g_dtau%g_ground.npy" % (N, bond_dim, J,h, tol_im, step_im)
dyn_fname = "t_ising_N%d_D%d_J%g_h%g_s%g_dtau%g_dt%g_tf%g_Jquench%g_hquench_%g" % \
			(N, bond_dim,J, h, tol_im, step_im,realstep,tf,J_quench,h_quench)

'''
A nearest-neighbour Hamiltonian is a sequence of 4-dimensional arrays, one for
each pair of sites.
For each term, the indices 0 and 1 are the 'bra' indices for the first and
second sites and the indices 2 and 3 are the 'ket' indices:

  ham[n][s,t,u,v] = <st|h|uv> (for sites n and n+1)
'''
def get_hamiltonian(num_sites,J,h):
	ham = (-J*sp.kron(Sz,Sz) - h*sp.kron(Sx,sp.eye(2))).reshape(2,2,2,2)
	return [ham]*num_sites




#Create vectors for bond dimension and local site hilber space
v_D = [bond_dim]*(N+1)
v_q = [2]*(N+1)
start_time = time.time()
#Create instance of evoMPS_generic class
sim = tdvp.EvoMPS_TDVP_Generic(N, v_D, v_q, get_hamiltonian(N, J,h))
sim.zero_tol = zero_tol
sim.sanity_checks = sanity_checks

#if ground state is already calculated then go to real time evolution otherwise first calculate ground state.
try:
	a_file = open("DQPT/Logs/"+grnd_fname, 'rb')
	sim.load_state(a_file)
	a_file.close
	groun_loaded = True
	print('Using saved ground state: ' + grnd_fname)
except IOError as e:
	groun_loaded = False
	print('No existing ground state could be opened.')

#if ground state is not loaded
if not groun_loaded:
	tg.t_ising_ground_calc(sim,grnd_fname, tol_im, step_im,auto_truncate=auto_truncate)
	a_file = open("DQPT/Logs/"+grnd_fname, 'rb')
	sim.load_state(a_file)
	a_file.close
	groun_loaded = True

#Start real time evolution
#sim.update(auto_truncate=auto_truncate)
#sim.update(auto_truncate=auto_truncate)

#grnd_state = sim.A.copy()	#Ground state

print ("Ground state energy before quench = ", sim.H_expect.real)
#print ("Ground state norm = ",matmul.contract(grnd_state,grnd_state))
print ("Maximum Bond dimension", sim.D)
T = []				#time vector
v_P = []			#Return probability vector (Loschmidt echo)
v_H = [] 			#Eergy vector
v_rate = []			#Rate function for Loschmidt echo
v_entropy = []		#vector to store Von Neumann Entropy
v_schmidt_sq = []	#store squared singular values at middle bond
t = 0.
eta = 1.
#ham_quench = get_hamiltonian(N,J_quench,h_quench)
sim.ham = get_hamiltonian(N,J_quench,h_quench)
sim.update(auto_truncate=auto_truncate)

print ("Starting real time evolution")
print()
col_heads = ["step","t","<H>","dH","prob","rate","norm of state"] #These last three are for testing the midpoint method.
print ("\t".join(col_heads))
print()
print('*********************************')
print(sim.r[-1])
print('*********************************')
for i in range(0,num_step):
	
	T.append(t)
	sim.update(auto_truncate=auto_truncate)

	v_H.append(sim.H_expect.real)

	if len(v_H) > 1 :
		dH = v_H[-1] - v_H[-2]
	else:
		dH = 0.

	'''
		Compute observable quantities
	'''
	#compute return probability
	#prob = abs(matmul.contract(grnd_state,sim.A).real)**2
	#compute rate function
	#r = (1./N)*sp.log(1./prob)
	#Compute Von-Neumann Entropy and singular values method from mps_gen
	#[ent,sv] = sim.entropy(N//2, ret_schmidt_sq=True)
	#norm = abs(matmul.contract(sim.A,sim.A).real)**2
	'''
		Store the variables
	'''
	#v_P.append(prob)
	#v_rate.append(r)
	#v_entropy.append(ent)
	#v_schmidt_sq.append(sv)

	#make arrangments for printing
	row = [str(i)]
	row.append(str(t))
	row.append("%.15g" % v_H[-1])
	row.append("%.8g" % dH)
	#row.append("%.15g" % prob)
	#row.append("%.15g" % r)
	#row.append("%.8g"%norm)


	#dynamic expansion before Rk4 step
	dstep = realstep**(5./2.)
	sim.take_step(dstep * 1.j, dynexp=True, dD_max=16, sv_tol=1E-5,D_max=bond_dim)
	err = sim.etaBB.real.sum()
	sim.update(auto_truncate=auto_truncate)
	dstep = 0.
	sim.take_step_RK4((realstep - dstep) * 1.j)
	t = t + realstep

	eta = sim.eta.real.sum() 

	if math.fmod(i,10) == 0:
		print ("\t".join(row))


print ("real time evolution for" , tf, "seconds completed !")
end_time = time.time()
print("Time taken for simulation %g seconds"%(end_time - start_time))
'''
	Plot the results
'''
if plot_results == True:
	fig1 = plt.figure(1)
	H_t = fig1.add_subplot(111)
	H_t.set_xlabel('time in seconds')
	H_t.set_ylabel('Energy')
	H_t.plot(T,v_H)

	fig2 = plt.figure(2)
	P_t = fig2.add_subplot(111)
	P_t.set_xlabel('time in seconds')
	P_t.set_ylabel('probability')
	P_t.set_title('Return Probabiliy')
	P_t.plot(T,v_P)

	fig3 = plt.figure(3)
	R_t = fig3.add_subplot(111)
	R_t.set_xlabel('time in seconds')
	R_t.set_ylabel('Return rate')
	R_t.set_title('-(1/N)log(P)')
	R_t.plot(T,v_rate)

	plt.show()





if save_data == True:
	mydir = os.path.join(os.getcwd(), "DQPT/Logs/","N_%g_D_%g"%(N,bond_dim),datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
	os.makedirs(mydir)
	print("saving data files")
	sp.save(mydir+"/"+grnd_fname+"_ground.npy", grnd_state)
	sp.save(mydir+"/"+dyn_fname+"_energy.npy", v_H)
	sp.save(mydir+"/"+dyn_fname+"_prob.npy", v_P)
	sp.save(mydir+"/"+dyn_fname+"_rate.npy", v_rate)
	sp.save(mydir+"/"+dyn_fname+"_time.npy",T)
	sp.save(mydir+"/"+dyn_fname+"_sim_time.npy",(end_time - start_time))
	sp.save(mydir+"/"+dyn_fname+"_entropy.npy", v_entropy)
	sp.save(mydir+"/"+dyn_fname+"_sv_sq.npy", v_schmidt_sq)
	print("Data files saved!!")
print("simulation done!!")
