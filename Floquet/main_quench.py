'''
	Main code file for quench dynamics
'''
import sys
import numpy as np
import matplotlib.pyplot as plt
import methods as func
import time
import os,datetime
import math


#parameters for simulation
L = 256
dt = 0.1
tf = 100.

A = 0.5
w = np.pi/4.
J = 0.5
mu = 1.
start_time = time.time()

initState = "ALT"
save_data = False	
save_terminal = False
plot_results = True
loaded  = False

calc_entropy = True
calc_nbar = True
calc_dist = False
calc_I = True
#subsystem size
l0 = L//2 - 5
l1 = L//2 + 5

fname = initState+"VecP_N_%d_J_%g_A%g_w%g_mu%g_tf_%g_dt_%g_"%(L,J,A,w,mu,tf,dt)
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
print("frequency of AC electric field = ",w)
print("Amplitude of AC electric field = ",A)
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

#Check if density matrix data already exists 
#load saved data otherwise compute using time evolution


N = int(tf/dt)
T = [dt*n for n in range(N)]
'''
try:
	a_file = open("Logs/"+fname+"CC.npy",'rb')
	CC_t = np.load(a_file)
	a_file.close
	loaded = True
	print("Using saved CC matrices : Logs/"+fname+"CC.npy")
except IOError as e:
	loaded = False
	print("No existing data file could be opened!!")

'''
	
if loaded == False:
	CC_t = []
	CC_t.append(CC_0)
	UU = np.zeros((L,L)) 
	CC_prev = CC_0.copy()
	for i in range(N-1):
		t = T[i]
		if math.fmod(i,100) == 0: 
			print(t)
		HH = func.getHam_Bfield_PBC(L,J,A,w,t,mu,sanity_checks=True)
		
		eps , DD = np.linalg.eigh(HH)
		EE = np.diag(np.exp(-1j*eps*dt))	#exp(-i_epsk_t)
		UU = np.dot(np.conj(DD),np.dot(EE,DD.T))	#derived
		CC_t.append(np.dot(np.conj(UU).T, np.dot(CC_prev, UU)) )
		CC_prev = CC_t[-1].copy()
	
	#if save_data == True:
		#np.save("Logs/"+fname+"CC.npy",CC_t)
		
		#print("Correlation matrices saved to : "+ "Logs/"+fname+"CC.npy")


#Calculate required observables from the data
if calc_entropy == True:

	S = []	#entropy time series
	eigCC = []	#spectrum of subsystem density matrix
	print('--------------------------------------------------------')
	print("Calculating vonNeumann Entropy")
	print()
	print('susbsytem from ',l0 ,' to ',l1,' sites of length ',l1-l0)
	print()
	for j in range(N):
		s , eigV = func.vonNeumannEntropy(CC_t[j],l0,l1)
		eigCC.append(eigV)
		S.append(s)

	#print("Equillibrium Entropy : ")
	if plot_results == True:

		plt.plot(T,S)
		#plt.axhline(y=np.log(2**(l1-l0)),color="r",linestyle='--')
		plt.xlabel('Time')
		plt.ylabel("entropy")
		plt.title("Entropy dynamics for electric field A*cos(wt) N = %d sites"%(L))
		plt.show()
	if save_data == True:
		
		np.save(mydir+"/"+fname+"entropy.npy",S)
		
		np.save(mydir+"/"+fname+"CCspectrum.npy",eigCC)

if calc_nbar == True:
	nbar = []	#average occupation in susystem
	for i in range(N):
		nn = 0.
		for j in range(l0,l1):
			nn = nn + CC_t[i][j,j].real
		nbar.append(nn/(l1-l0))
	print()
	print('Local occupation number calculated')
	print()
	if plot_results == True:

		plt.plot(T,nbar)
		plt.xlabel('Time')
		plt.ylabel("local occupation number ")
		plt.title("Local occupation number for N = %d sites"%(L),loc='right')
		plt.show()
	if save_data == True:
		
		np.save(mydir+"/"+fname+"nbar.npy",nbar)
		
	
if calc_dist == True:
	dd = []
	for k in range(N):
		dd.append(func.distance(CC_t[k][l0:l1],CC_t[-1][l0:l1]))
	print()
	print('Distance measure calculated')
	print()
	if save_data == True:
		np.save(mydir+"/"+fname+"distance.npy",dd)

if calc_I == True:
	II = []
	index_o = range(0,L,2)
	index_e = range(1,L,2)
	for k in range(N):
		diag = CC_t[k].diagonal()
		n_o = sum(diag[index_o])
		n_e = sum(diag[index_e])
		II.append((n_o + n_e)/(n_o - n_e))
	print()
	print('Inversion in number calculated')
	print()
	if save_data == True:
		np.save(mydir+"/"+fname+"inv.npy",II)
		
end_time = time.time()
print('--------------------------------------------------------')
print("Simulation Over!! Time taken : " , end_time - start_time," seconds")
if save_data == True and save_terminal==True:
	f.close()
