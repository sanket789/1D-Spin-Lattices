'''
	Main code file for quench dynamics. This code doesn't save correlation matrices
'''
import sys
import numpy as np
import matplotlib.pyplot as plt
import methods as func
import time
import os,datetime
import math


#parameters for simulation

L = 100			#Length of chain. sites from 0 , 1 , ... , L-1
dt = 0.1 		#time step
tf = 100. 		#total time

#Paramteres for Hamiltonian 
A = 0.0			#Amplitude of Electric field (in V per lattice constant)

w = np.pi 		#Frequency of time dep field
J = 0.5			#hopping parameter for TB chain
mu0 = 1. 		#chemical potential
sigma = 0.5*(np.sqrt(5) - 1)	#irrational number for quasi-periodicity
alpha = 0. 		#random num

start_time = time.time()

initState = "ALT" 	#initial state
save_data = False	
save_terminal = False
plot_results = True
loaded  = False

#subsystem size (l0 to l1-1)
l0 = L//2 - 5
l1 = L//2 + 5

#create filename for storage
fname = initState+"FLUX_N_%d_J_%g_w_%g_mu0_%g_s_%g_al_%g_tf_%g_dt_%g_"%(L,J,w,mu0,sigma,alpha,tf,dt)
if save_data == True:
	mydir = os.path.join(os.getcwd(), "Logs/","N_%g_w_%g"%(L,w),datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
	os.makedirs(mydir)
	if save_terminal == True:
		f = open(mydir+"/"+"terminal.out", 'w')
		sys.stdout = f
	

#Define initial density matrix CC_0
CC_0 = np.zeros((L,L))
print('--------------  Simulation for  ----------------')
print("Number of sites = ", L)
print("frequency of Drive = ",w)
#print("Amplitude of AC electric field = ",A)
print("Chemical potential = %gcos(2pi*n*%g + %g)"%(mu0,sigma,alpha))

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

#Check if correlation matrix data already exists 
#load saved data otherwise compute using time evolution


N = int(tf/dt)	#number of time steps
T = [dt*n for n in range(N)]	#list of time
try:
	a_file = open("Logs/"+fname+"CC.npy",'rb')
	CC_t = np.load(a_file)
	a_file.close
	loaded = True
	print("Using saved CC matrices : Logs/"+fname+"CC.npy")
except IOError as e:
	loaded = False
	print("No existing data file could be opened!!")

UU = np.zeros((L,L)) 
CC_curr = CC_0.copy()	
S = []					#entropy time series
eigCC = []				#spectrum of subsystem density matrix
nbar = []				#average occupation in subsystem
II = []					#even-odd diff in subsystem
JJ_charge = []
index_o = range(0,L,2)	#odd indices
index_e = range(1,L,2)	#evevn indices

for i in range(N):
	t = T[i]
	if loaded == False:
		HH = func.getHam_flux_PBC(L,J,w,t,mu0,sigma,alpha,sanity_checks=True)			#get Hamiltonian
		
		eps , DD = np.linalg.eigh(HH)							#Find eigen values and eigenvevtors
		EE = np.diag(np.exp(-1j*eps*dt))						#exp(-i_epsk_t)
		UU = np.dot(np.conj(DD),np.dot(EE,DD.T))				#derived
		CC_next = (np.dot(np.conj(UU).T, np.dot(CC_curr, UU)) )	#calculate CC for t+dt
	else:
		CC_curr = CC_t[i].copy()
	
	#Calculate entropy 
	s , eigV = func.vonNeumannEntropy(CC_curr,l0,l1)
	eigCC.append(eigV)
	S.append(s)
	
	diag = CC_curr.diagonal().real

	#Caluculate nbar 
	nbar.append(sum(diag[l0:l1]).real/(l1-l0))
	
	#Calculate I
	n_o = sum(diag[index_o])
	n_e = sum(diag[index_e])
	II.append((n_o + n_e)/(n_o - n_e))
	
	#Calculate charge current
	JJ_charge.append(func.getCurrent(CC_curr,w,t,J,"FLUX"))
	
	if loaded == False:
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
		
if save_data == True:
	
	np.save(mydir+"/"+fname+"entropy.npy",S)
	
	np.save(mydir+"/"+fname+"CCspectrum.npy",eigCC)
	np.save(mydir+"/"+fname+"nbar.npy",nbar)
	np.save(mydir+"/"+fname+"inv.npy",II)
		
	print("Saved data files to ", mydir )
end_time = time.time()
print('--------------------------------------------------------')
print("Simulation Over!! Time taken : " , end_time - start_time," seconds")
if save_data == True and save_terminal==True:
	f.close()
