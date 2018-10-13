'''
Main code for simulating fermion tight binding chain in presence of external drive.
The chemical potential is site dependent quasiperiodic function
The correlation matrices are averaged over alpha parmater (phase of quasiperiodic function)
'''
import sys
import numpy as np
import matplotlib.pyplot as plt
import methods as func
import time
import os,datetime
import math
from mpi4py import MPI
import argparse


def main_routine(arg,c,start_time):
	mpi_rank = c.Get_rank()
	mpi_size = c.Get_size()


	#parameters for simulation
	L = arg.L				#Length of chain. sites from 0 , 1 , ... , L-1
	dt = 0.1 			#time step
	tf = 10 			#total time
	NUM_CONF = arg.NUM_CONF 	#Number of samples for alpha Always a multiple of mpi_size
	#Paramteres for Hamiltonian 
	A = arg.A				#Amplitude of drive
	w = arg.w*np.pi 		#Frequency of time dep field
	J = 1.0				#hopping parameter for TB chain
	mu0 = arg.mu0 			#chemical potential

	sigma = 0.5*(np.sqrt(5) + 1)	#irrational number for quasi-periodicity

	initState = "ALT" 	#initial state
	save_data = True	
	plot_results = False

	#subsystem size (l0 to l1-1)
	if L>20:
		l0 = L//2 - 5
		l1 = L//2 + 5
	else:
		l0 = 0
		l1 = L//2
	#create filename for storage
	fname = initState+"_AVG_N_%d_J_%g_A_%g_w_%g_mu0_%g_s_%g_tf_%g_dt_%g_"%(L,J,A,w,mu0,sigma,tf,dt)
	
	print('--------------  Simulation for  ----------------')
	print("Number of sites = ", L)
	print("frequency of Drive = ",w)
	print("Amplitude of Drive = ",A)
	print("Chemical potential = %gcos(2pi*n*%g + '<alpha>')"%(mu0,sigma))


	CC_0 = np.zeros((L,L),dtype=complex)	#Define initial density matrix CC_0
	if initState == "ALT":
		print("Using half filled state with alternate occupancy | 1 0 1 0 1 ...>")
		for k in range(0,L,2):
			CC_0[k,k] = 1.0

	elif initState=="LH":
		print("Using half filled state with left half occupancy | 11111 ...0000>")
		for i in range(0,L//2):
			CC_0[i,i] = 1.
	else:  
		exitmsg = "Use appropriate intial state"
		sys.exit(s)

	print("--------------------------------------------------------")
	print()

	'''
	Some declarations and initializations
	'''

	alpha_min = mpi_rank*2*np.pi/mpi_size
	alpha_max = (mpi_rank+1)*2*np.pi/mpi_size
	num_alpha = NUM_CONF//mpi_size

	N = int(tf/dt)	#number of time steps
	T = [dt*n for n in range(N)]	#list of time
		
	S = np.zeros((num_alpha,N))					#entropy time series
	eigCC = np.zeros((num_alpha,N,l1-l0))		#spectrum of subsystem density matrix
	nbar = 	np.zeros((num_alpha,N))				#average occupation in subsystem
	II = np.zeros((num_alpha,N))				#even-odd diff in subsystem
	JJ_charge = np.zeros((num_alpha,N,L))		#charge current
	index_o = range(0,L,2)						#odd indices
	index_e = range(1,L,2)						#even indices
	E_expect = np.zeros((num_alpha,N))			#Energy expectation values
	nn_site = np.zeros((num_alpha,N))			#number operator one site L/2
	conf = np.linspace(alpha_min,alpha_max,num_alpha,endpoint=False)					#disorder parameters
	
	for k in range(num_alpha):
		UU = np.zeros((L,L),dtype=complex) 
		CC_curr = CC_0.copy()	
		for i in range(N):
			t = T[i]
			HH = func.getHam_flux_PBC(L,J,w,t,mu0,sigma,conf[k],A,sanity_checks=False)			#get Hamiltonian
			
			#Calculate entropy 
			s , eigV = func.vonNeumannEntropy(CC_curr,l0,l1)
			eigCC[k,i,:] = eigV.copy() 	#eigenvalues of subsystem correlation matrix
			S[k,i] = s
			
			diag = CC_curr.diagonal().real

			#Calculate nbar 
			nbar[k,i] = np.sum(diag[l0:l1]).real/(l1-l0)
			
			#Calculate I
			n_o = np.sum(diag[index_o])
			n_e = np.sum(diag[index_e])
			II[k,i] = ((n_o - n_e)/(n_o + n_e))
			
			#Calculate charge current
			JJ_charge[k,i,:] = func.getCurrent(CC_curr,w,t,J,"FLUX")

			#Calculate Energy expectation value
			E_expect[k,i] = np.sum(np.multiply(HH,CC_curr)).real
			
			#Calculate number operator at L/2
			nn_site[k,i] = CC_curr[L//2,L//2].real

			#Update CC for next iteration
			eps , DD = np.linalg.eigh(HH)							#Find eigen values and eigenvevtors
			EE = np.diag(np.exp((-1j*dt)*eps))						#exp(-i_epsk_t)
			UU = np.dot(np.conj(DD),np.dot(EE,DD.T))				#derived
			CC_next = (np.dot(np.conj(UU).T, np.dot(CC_curr, UU)) )	#calculate CC for t+dt
			CC_curr = CC_next.copy()
		print('Done alpha = ', conf[k] )	

	'''
		To gather the calculated data
	'''
	recv_S = None
	recv_eigCC = None
	recv_nbar = None
	recv_II = None
	recv_JJ_charge = None
	recv_E_expect = None
	recv_nn_site = None
	recv_conf = None

	if mpi_rank == 0:
		recv_S = np.empty([mpi_size,num_alpha,N])
		recv_eigCC = np.empty([mpi_size,num_alpha,N,l1-l0])
		recv_nbar = np.empty([mpi_size,num_alpha,N])
		recv_II = np.empty([mpi_size,num_alpha,N])
		recv_JJ_charge = np.empty([mpi_size,num_alpha,N,L])
		recv_E_expect = np.empty([mpi_size,num_alpha,N])
		recv_nn_site = np.empty([mpi_size,num_alpha,N])
		recv_conf = np.empty([mpi_size,num_alpha])
	
	## saving 
	## Entanglement, eigenvalues (sub-system), nbar(sub), nbar(1-body), imbalance, current, energy)

	comm.Gather(S,recv_S,root=0)
	comm.Gather(eigCC, recv_eigCC,root=0)
	comm.Gather(nbar,recv_nbar,root=0)
	comm.Gather(II,recv_II,root=0)
	comm.Gather(JJ_charge,recv_JJ_charge,root=0)
	comm.Gather(E_expect,recv_E_expect,root=0)
	comm.Gather(nn_site,recv_nn_site,root=0)
	comm.Gather(conf,recv_conf,root=0)

	
	if plot_results == True and mpi_rank == 0:

		plt.plot(T,S)
		#plt.axhline(y= np.log(2.**(l1-l0)),color="r",linestyle='--')
		plt.xlabel('Time')
		plt.ylabel("entropy")
		plt.title("Entropy dynamics for electric field A*cos(wt) N = %d sites"%(L))
		plt.show()
		
		plt.plot(T,nbar)
		plt.xlabel('Time')
		plt.ylabel("local occupation number ")
		plt.title("Local occupation number for N = %d sites"%(L),loc='right')
		plt.show()
		
		plt.plot(range(L),JJ_charge[-1])

		plt.xlabel('Sites')
		plt.ylabel("Final current ")
		plt.title("curent at time %g for N = %d sites"%(tf,L),loc='right')
		plt.show()

		plt.plot(T,E_expect)
		plt.xlabel('Time')
		plt.ylabel('Energy expectation value')
		plt.show()
			
	if mpi_rank ==0:
		#averaging over disorder configurations
		# fin_S = np.average(recv_S,axis=0)
		# fin_eigCC = np.average(recv_eigCC,axis=0)
		# fin_nbar = np.average(recv_nbar,axis=0)
		# fin_II = np.average(recv_II,axis=0)
		# fin_JJ_charge = np.average(recv_JJ_charge,axis=0)
		# fin_E_expect = np.average(recv_E_expect,axis=0)
		# fin_nn_site = np.average(recv_nn_site,axis=0)

		if save_data == True:
			mydir = os.path.join(os.getcwd(), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
			
			os.makedirs(mydir)
			
			np.save(mydir+"/"+fname+"entropy.npy",recv_S)
			np.save(mydir+"/"+fname+"CCspectrum.npy",recv_eigCC)
			np.save(mydir+"/"+fname+"nbar.npy",recv_nbar)
			np.save(mydir+"/"+fname+"inv.npy",recv_II)
			np.save(mydir+"/"+fname+"current.npy",recv_JJ_charge)
			np.save(mydir+"/"+fname+"E_expect.npy",recv_E_expect)
			np.save(mydir+"/"+fname+"nn_site.npy",recv_nn_site)
			np.save(mydir+"/"+fname+"conf.npy",recv_conf)

			print("Saved data files to ", mydir )
		end_time = time.time()
		print('--------------------------------------------------------')
		print("Simulation Over!! Time taken : " , end_time - start_time," seconds")
	

if __name__ == '__main__':
	start_time = time.time()
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	size = comm.Get_size()

	if rank==0:
		## parsing at rank =0
		parser = argparse.ArgumentParser(
				  description='Time evolution of fermion chain',
				  prog='main', usage='%(prog)s [options]')
		parser.add_argument("-L", "--L", help="System Size",default=512,type=int, nargs='?')
		parser.add_argument("-w", "--w", help="Frequency of drive",default=1.0,type=float, nargs='?')
		parser.add_argument("-mu0", "--mu0", help="Strength of chemical potential",default=2.0,type=float, nargs='?')
		parser.add_argument("-A", "--A", help="Strength of drive",default=1.0,type=float, nargs='?')
		parser.add_argument("-NUM_CONF","--NUM_CONF",help = "number of disorder config",default=100,type=int,nargs='?')
		#-- add in the argument
		args=parser.parse_args()

	else:
		args = None

	# broadcast
	args = comm.bcast(args, root=0)

	# run main code
	main_routine(args, comm,start_time)



