import numpy as np
import matplotlib.pyplot as plt
import os

N0 = 512
N1 = 256

NUM = [N0,N1]
J = 0.5
dt = 0.1
tf = 1000.
A = 0.5
mu = 1.
N = int(tf/dt)
T = [dt*n for n in range(N)]

w0 = np.pi
w1 = np.pi/2.
w2 = np.pi/4.
w3 = np.pi/8.

w = [w0,w1,w2,w3]

mydir = [[],[]]
fname = [[],[]]
folder =[	["2018-09-16_23-07-46" , "2018-09-16_23-07-52" , "2018-09-16_23-07-59" , "2018-09-16_23-08-03" ],
			["2018-09-16_23-07-31" , "2018-09-16_23-07-16" , "2018-09-16_23-07-07" , "2018-09-16_22-58-17" ]]

for i in range(2):
	for j in range(4):

		mydir[i].append(os.path.join(os.getcwd(), "Logs/","N_%g_w_%g"%(NUM[i],w[j]),folder[i][j]))
		fname[i].append('ALT'+"VecP_N_%d_J_%g_A%g_w%g_mu%g_tf_%g_dt_%g_"%(NUM[i],J,A,w[j],mu,tf,dt))


for i in range(2):


	#plot nbar
	NB0 = np.load(mydir[i][0]+"/"+fname[i][0]+"nbar.npy")
	NB1 = np.load(mydir[i][1]+"/"+fname[i][1]+"nbar.npy")
	NB2 = np.load(mydir[i][2]+"/"+fname[i][2]+"nbar.npy")
	NB3 = np.load(mydir[i][3]+"/"+fname[i][3]+"nbar.npy")
	#NB4 = np.load(mydir[i][4]+"/"+fname[i][4]+"nbar.npy")	

	plt.plot(T,NB0.real,label="w = %g"%(w[0]),color="b")
	plt.plot(T,NB1.real,label="w = %g"%(w[1]),color="k")
	plt.plot(T,NB2.real, label="w = %g"%(w[2]),color="c")
	plt.plot(T,NB3.real, label="w = %g"%(w[3]),color="g")
	#plt.plot(T,NB4.real, label="w = %g"%(w[4]),color="r")
	plt.legend(loc=0,prop={'size': 10})
	plt.xlabel('Time')
	plt.ylabel("local occupation number ")
	plt.title("Local occupation number for N = %d sites"%(NUM[i]),loc='right')
	plt.savefig("Logs/plots/nbar_N_%d"%(NUM[i]))
	
	plt.show()
	'''
	#plot distance
	D0 = np.load(mydir[i][0]+"/"+fname[i][0]+"distance.npy")
	D1 = np.load(mydir[i][1]+"/"+fname[i][1]+"distance.npy")
	D2 = np.load(mydir[i][2]+"/"+fname[i][2]+"distance.npy")
	D3 = np.load(mydir[i][3]+"/"+fname[i][3]+"distance.npy")
	#D4 = np.load(mydir[i][4]+"/"+fname[i][4]+"distance.npy")	

	plt.plot(T,D0,label="w = %g"%(w[0]),color="b")
	plt.plot(T,D1,label="w = %g"%(w[1]),color="k")
	plt.plot(T,D2, label="w = %g"%(w[2]),color="c")
	plt.plot(T,D3, label="w = %g"%(w[3]),color="g")
	#plt.plot(T,D4, label="w = %g"%(w[4]),color="r")
	plt.legend(loc=3,prop={'size': 10})
	plt.xlabel('Time')
	plt.ylabel("Distance Measure")
	plt.title("Distance between C and C_eq  for N = %d sites"%(NUM[i]))
	plt.savefig("Logs/dplots/istance_N_%d"%(NUM[i]))
	
	plt.show()
	'''

	#plot entropy

	S0 = np.load(mydir[i][0]+"/"+fname[i][0]+"entropy.npy")
	S1 = np.load(mydir[i][1]+"/"+fname[i][1]+"entropy.npy")
	S2 = np.load(mydir[i][2]+"/"+fname[i][2]+"entropy.npy")
	S3 = np.load(mydir[i][3]+"/"+fname[i][3]+"entropy.npy")
	#S4 = np.load(mydir[i][4]+"/"+fname[i][4]+"entropy.npy")

	plt.plot(T,S0,label="w = %g"%(w[0]),color="b")
	plt.plot(T,S1,label="w = %g"%(w[1]),color="k")
	plt.plot(T,S2, label="w = %g"%(w[2]),color="c")
	plt.plot(T,S3, label="w = %g"%(w[3]),color="g")
	#plt.plot(T,S4, label="w = %g"%(w[4]),color="r")
	plt.axhline(y= np.log(2.**(10.)),color="r",linestyle='--')
	plt.legend(loc=5,prop={'size': 10})
	plt.xlabel('Time')
	plt.ylabel("entropy")
	plt.title("Entropy dynamics for electric field A*cos(wt) N = %d sites"%(NUM[i]))
	plt.savefig("Logs/plots/entropy_N_%d"%(NUM[i]))
	
	plt.show()
	
	#plot II
	I0 = np.load(mydir[i][0]+"/"+fname[i][0]+"inv.npy")
	I1 = np.load(mydir[i][1]+"/"+fname[i][1]+"inv.npy")
	I2 = np.load(mydir[i][2]+"/"+fname[i][2]+"inv.npy")
	I3 = np.load(mydir[i][3]+"/"+fname[i][3]+"inv.npy")
	#I4 = np.load(mydir[i][4]+"/"+fname[i][4]+"inv.npy")

	plt.plot(T,1./I0.real,label="w = %g"%(w[0]),color="b")
	plt.plot(T,1./I1.real,label="w = %g"%(w[1]),color="k")
	plt.plot(T,1./I2.real, label="w = %g"%(w[2]),color="c")
	plt.plot(T,1./I3.real, label="w = %g"%(w[3]),color="g")
	#plt.plot(T,1./I4.real, label="w = %g"%(w[4]),color="r")
	plt.legend(loc=5,prop={'size': 10})
	plt.xlabel('Time')
	plt.ylabel("Odd-even Difference")
	plt.title("(N_odd - N_even)/N for N = %d sites"%(NUM[i]))
	plt.savefig("Logs/plots/II_N_%d"%(NUM[i]))
	
	plt.show()
	
	if i==0:
		
		plt.subplot(2,2,1)
		plt.plot(T,1./I0.real,label="w = %g"%(w[0]),color="b")
		plt.legend(loc=5,prop={'size': 10})
		plt.xlabel('Time')
		plt.ylabel("Odd-even Difference")
		
		
		plt.subplot(2,2,2)
		plt.plot(T,1./I1.real,label="w = %g"%(w[1]),color="k")
		plt.legend(loc=5,prop={'size': 10})
		plt.xlabel('Time')
		plt.ylabel("Odd-even Difference")
		
		plt.subplot(2,2,3)
		plt.plot(T,1./I2.real, label="w = %g"%(w[2]),color="c")
		plt.legend(loc=5,prop={'size': 10})
		plt.xlabel('Time')
		plt.ylabel("Odd-even Difference")
		
		plt.subplot(2,2,4)
		plt.plot(T,1./I3.real, label="w = %g"%(w[3]),color="g")
		plt.legend(loc=5,prop={'size': 10})
		plt.xlabel('Time')
		plt.ylabel("Odd-even Difference")
		
		plt.suptitle("(N_odd - N_even)/N for N = %d sites"%(NUM[i]))
		plt.savefig("Logs/plots/subII_N_%d"%(NUM[i]))
		
		plt.show()
		
		plt.subplot(2,2,1)
		plt.plot(T,S0,label="w = %g"%(w[0]),color="b")
		plt.axhline(y= np.log(2.**(10.)),color="r",linestyle='--')
		plt.legend(loc=5,prop={'size': 10})
		plt.xlabel('Time')
		plt.ylabel("Entropy")
		
		plt.subplot(2,2,2)
		plt.plot(T,S1,label="w = %g"%(w[1]),color="k")
		plt.axhline(y= np.log(2.**(10.)),color="r",linestyle='--')
		plt.legend(loc=5,prop={'size': 10})
		plt.xlabel('Time')
		plt.ylabel("Entropy")
		
		plt.subplot(2,2,3)
		plt.plot(T,S2, label="w = %g"%(w[2]),color="c")
		plt.axhline(y= np.log(2.**(10.)),color="r",linestyle='--')
		plt.legend(loc=5,prop={'size': 10})
		plt.xlabel('Time')
		plt.ylabel("Entropy")
		
		plt.subplot(2,2,4)
		plt.plot(T,S3, label="w = %g"%(w[3]),color="g")
		plt.axhline(y= np.log(2.**(10.)),color="r",linestyle='--')
		plt.legend(loc=5,prop={'size': 10})
		plt.xlabel('Time')
		plt.ylabel("Entropy")
		
		plt.suptitle("Entropy for N = %d sites"%(NUM[i]))
		plt.savefig("Logs/plots/subEnt_%d"%(NUM[i]))
		
		plt.show()
