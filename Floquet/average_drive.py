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

#parameters for simulation

L = 512				#Length of chain. sites from 0 , 1 , ... , L-1
dt = 0.125 			#time step
tf = 1000. 			#total time
avg_sample = 100 	#Number of samples for alpha
#Paramteres for Hamiltonian 
A = 2.0*L				#Amplitude of drive
w = 16. 		#Frequency of time dep field
J = 1.0				#hopping parameter for TB chain
mu0 = 0.5 			#chemical potential

sigma = 0.5*(np.sqrt(5) + 1)	#irrational number for quasi-periodicity

start_time = time.time()

initState = "ALT" 	#initial state
save_data = True	
save_terminal = True
plot_results = False

#subsystem size (l0 to l1-1)
l0 = L//2 - 5
l1 = L//2 + 5

#create filename for storage
fname = initState+"_AVG_N_%d_J_%g_A_%g_w_%g_mu0_%g_s_%g_tf_%g_dt_%g_"%(L,J,A,w,mu0,sigma,tf,dt)
if save_data == True:
	mydir = os.path.join(os.getcwd(), "Logs/","N_%g_w_%g"%(L,w),datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
	os.makedirs(mydir)
	if save_terminal == True:
		f = open(mydir+"/"+"terminal.out", 'w')
		sys.stdout = f

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
N = int(tf/dt)	#number of time steps
T = [dt*n for n in range(N)]	#list of time
	
S = np.zeros((N,1))					#entropy time series
eigCC = np.zeros((N,l1-l0))				#spectrum of subsystem density matrix
nbar = 	np.zeros((N,1))				#average occupation in subsystem
II = np.zeros((N,1))				#even-odd diff in subsystem
JJ_charge = np.zeros((N,L))			#charge current
index_o = range(0,L,2)				#odd indices
index_e = range(1,L,2)				#even indices
E_expect = np.zeros((N,1))			#Energy expectation values
nn_site = np.zeros((N,1))			#number operator one site L/2

for alpha in np.linspace(0.0,2*np.pi,avg_sample):
	UU = np.zeros((L,L),dtype=complex) 
	CC_curr = CC_0.copy()	
	for i in range(N):
		t = T[i]
		HH = func.getHam_flux_PBC(L,J,w,t,mu0,sigma,alpha,A,sanity_checks=True)			#get Hamiltonian
		
		#Calculate entropy 
		s , eigV = func.vonNeumannEntropy(CC_curr,l0,l1)
		eigCC[i,:] = eigCC[i,:]+ (1./avg_sample)*eigV
		S[i] += (1./avg_sample)*s
		
		diag = CC_curr.diagonal().real

		#Calculate nbar 
		nbar[i] += (1./avg_sample)*np.sum(diag[l0:l1]).real/(l1-l0)
		
		#Calculate I
		n_o = np.sum(diag[index_o])
		n_e = np.sum(diag[index_e])
		II[i] = (1./avg_sample)*((n_o + n_e)/(n_o - n_e))
		
		#Calculate charge current
		JJ_charge[i,:] += (1./avg_sample)*func.getCurrent(CC_curr,w,t,J,"FLUX")

		#Calculate Energy expectation value
		E_expect[i] += (1./avg_sample)*np.sum(np.multiply(HH,CC_curr)).real
		
		#Calculate number operator at L/2
		nn_site[i] += (1./avg_sample)*CC_curr[L//2,L//2].real

		#Update CC for next iteration
		eps , DD = np.linalg.eigh(HH)							#Find eigen values and eigenvevtors
		EE = np.diag(np.exp((-1j*dt)*eps))						#exp(-i_epsk_t)
		UU = np.dot(np.conj(DD),np.dot(EE,DD.T))				#derived
		CC_next = (np.dot(np.conj(UU).T, np.dot(CC_curr, UU)) )	#calculate CC for t+dt
		CC_curr = CC_next.copy()
		

if plot_results == True:

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
		
if save_data == True:
	
	np.save(mydir+"/"+fname+"entropy.npy",S)
	np.save(mydir+"/"+fname+"CCspectrum.npy",eigCC)
	np.save(mydir+"/"+fname+"nbar.npy",nbar)
	np.save(mydir+"/"+fname+"inv.npy",II)
	np.save(mydir+"/"+fname+"current.npy",JJ_charge)
	np.save(mydir+"/"+fname+"E_expect.npy",E_expect)
	np.save(mydir+"/"+fname+"nn_site.npy",nn_site)

	print("Saved data files to ", mydir )
end_time = time.time()
print('--------------------------------------------------------')
print("Simulation Over!! Time taken : " , end_time - start_time," seconds")
if save_data == True and save_terminal==True:
	f.close()


