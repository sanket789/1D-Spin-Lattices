'''
	Main code file for quench dynamics
'''
import sys
import numpy as np
import matplotlib.pyplot as plt
import methods as func
import time
import os,datetime



#parameters for simulation
NUM_SITES = 512	
L = int(0.5*NUM_SITES)
dt = 0.1
tf = 100.

A = 0.5
w = np.pi*2
J = 0.5

start_time = time.time()

initState = "ALT"
save_data = True	
plot_results = False
loaded  = False
calc_entropy = True
calc_nbar = True
calc_dist = True
l0 = L - 5
l1 = L + 5

fname = initState+"_N_%d_J_%g_A%g_w%g_tf_%g_dt_%g_"%(2*L,J,A,w,tf,dt)
if save_data == True:
	mydir = os.path.join(os.getcwd(), "Logs/","N_%g_w_%g"%(2*L,w),datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
	os.makedirs(mydir)
	f = open(mydir+"/"+"terminal.out", 'w')
	sys.stdout = f
	#with open(mydir+"/"+"terminal.out", 'w') as f:
	#	sys.stdout = f

#Define initial density matrix CC_0
CC_0 = np.zeros((2*L,2*L))
print('--------------  Simulation for  ----------------')
print("Number of sites = ", NUM_SITES)
print("frequency of AC electric field = ",w)
print("Amplitude of AC electric field = ",A)
if initState == "ALT":
	print("Using half filled state with alternate occupancy | 1 0 1 0 1 ...>")
	for k in range(0,2*L,2):
		CC_0[k,k] = 1.0

elif initState=="LH":
	print("Using half filled state with left half occupancy | 11111 ...0000>")
	for i in range(0,L):
		CC_0[i,i] = 1.
else:  
	exitmsg = "Use appropriate intial state"
	sys.exit(s)

print("-------------------------------------------------")
print()

#Check if density matrix data already exists 
#load saved data otherwise compute using time evolution


N = int(tf/dt)
T = [dt*n for n in range(N)]
try:
	a_file = open("Logs/"+fname+"CC.npy",'rb')
	CC_t = np.load(a_file)
	a_file.close
	loaded = True
	print("Using saved CC matrices : Logs/"+fname+"CC.npy")
except IOError as e:
	loaded = False
	print("No existing data file could be opened!!")


	
if loaded == False:
	CC_t = []
	UU = np.zeros((2*L,2*L)) 
	CC_prev = CC_0.copy()
	for i in range(N):
		t = T[i]
		HH = func.getHam(L,J,A,w,t)
		eps , DD = np.linalg.eigh(HH)
		EE = np.diag(np.exp(-1j*eps*dt))	#exp(-i_epsk_t)
		UU = np.dot(np.conj(DD),np.dot(EE,DD.T))	#derived
		CC_t.append(np.dot(np.conj(UU).T, np.dot(CC_prev, UU)) )
		CC_prev = CC_t[-1].copy()
	
	if save_data == True:
		np.save("Logs/"+fname+"CC.npy",CC_t)
		
		print("Correlation matrices saved to : "+ "Logs/"+fname+"CC.npy")


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

	print("Equillibrium Entropy : ")
	if plot_results == True:

		plt.plot(T,S)
		plt.show()
	if save_data == True:
		
		np.save(mydir+"/"+fname+"entropy.npy",S)
		
		np.save(mydir+"/"+fname+"CCspectrum.npy",eigCC)

if calc_nbar == True:
	nbar = []	#average occupation in susystem
	for i in range(N):
		nn = 0.
		for j in range(l0,l1):
			nn = nn + CC_t[i][j,j]
		nbar.append(nn/(l1-l0))
	print()
	print('Local occupation number calculated')
	print()
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


end_time = time.time()
print("Simulation Over!! Time taken : " , end_time - start_time," seconds")
f.close()
