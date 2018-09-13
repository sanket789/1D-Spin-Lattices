import numpy as np
import matplotlib.pyplot as plt
import os

N0 = 128
N1 = 256
N2 = 512
NUM = [N0,N1,N2]
J = 0.5
dt = 0.1
tf = 100.
A = 0.5

N = int(tf/dt)
T = [dt*n for n in range(N)]

w0 = 2.*np.pi

w1 = np.pi
w2 = np.pi/2.
w3 = np.pi/4.
w4 = np.pi/8.
w = [w0,w1,w2,w3,w4]

mydir = [[],[],[]]
fname = [[],[],[]]
folder =[	["2018-09-06_23-39-20" , "2018-09-06_23-39-33" , "2018-09-06_23-39-46" , "2018-09-06_23-40-00" , "2018-09-06_23-40-11"],
			["2018-09-06_23-40-28" , "2018-09-06_23-40-51" , "2018-09-06_23-42-07" , "2018-09-06_23-43-50" , "2018-09-06_23-46-13"],
			["2018-09-07_07-14-16" , "2018-09-07_07-14-54" , "2018-09-07_07-15-06" , "2018-09-07_07-15-16" , "2018-09-07_07-15-30"]	]

for i in range(3):
	for j in range(5):

		mydir[i].append(os.path.join(os.getcwd(), "Logs/","N_%g_w_%g"%(NUM[i],w[j]),folder[i][j]))
		fname[i].append('ALT'+"_N_%d_J_%g_A%g_w%g_tf_%g_dt_%g_"%(NUM[i],J,A,w[j],tf,dt))


for i in range(3):


	#plot distance
	NB0 = np.load(mydir[i][0]+"/"+fname[i][0]+"nbar.npy")
	NB1 = np.load(mydir[i][1]+"/"+fname[i][1]+"nbar.npy")
	NB2 = np.load(mydir[i][2]+"/"+fname[i][2]+"nbar.npy")
	NB3 = np.load(mydir[i][3]+"/"+fname[i][3]+"nbar.npy")
	NB4 = np.load(mydir[i][4]+"/"+fname[i][4]+"nbar.npy")	

	plt.plot(T,NB0.real,label="w = %g"%(w[0]),color="b")
	plt.plot(T,NB1.real,label="w = %g"%(w[1]),color="k")
	plt.plot(T,NB2.real, label="w = %g"%(w[2]),color="c")
	plt.plot(T,NB3.real, label="w = %g"%(w[3]),color="g")
	plt.plot(T,NB4.real, label="w = %g"%(w[4]),color="r")
	plt.legend(loc=0,prop={'size': 10})
	plt.xlabel('Time')
	plt.ylabel("local occupation number ")
	plt.title("Local occupation number for N = %d sites"%(NUM[i]),loc='right')
	plt.savefig("Logs/nbar_N_%d"%(NUM[i]))
	
	plt.show()
	
	#plot distance
	D0 = np.load(mydir[i][0]+"/"+fname[i][0]+"distance.npy")
	D1 = np.load(mydir[i][1]+"/"+fname[i][1]+"distance.npy")
	D2 = np.load(mydir[i][2]+"/"+fname[i][2]+"distance.npy")
	D3 = np.load(mydir[i][3]+"/"+fname[i][3]+"distance.npy")
	D4 = np.load(mydir[i][4]+"/"+fname[i][4]+"distance.npy")	

	plt.plot(T,D0,label="w = %g"%(w[0]),color="b")
	plt.plot(T,D1,label="w = %g"%(w[1]),color="k")
	plt.plot(T,D2, label="w = %g"%(w[2]),color="c")
	plt.plot(T,D3, label="w = %g"%(w[3]),color="g")
	plt.plot(T,D4, label="w = %g"%(w[4]),color="r")
	plt.legend(loc=3,prop={'size': 10})
	plt.xlabel('Time')
	plt.ylabel("Distance Measure")
	plt.title("Distance between C and C_eq  for N = %d sites"%(NUM[i]))
	plt.savefig("Logs/distance_N_%d"%(NUM[i]))
	
	plt.show()


	#plot entropy

	S0 = np.load(mydir[i][0]+"/"+fname[i][0]+"entropy.npy")
	S1 = np.load(mydir[i][1]+"/"+fname[i][1]+"entropy.npy")
	S2 = np.load(mydir[i][2]+"/"+fname[i][2]+"entropy.npy")
	S3 = np.load(mydir[i][3]+"/"+fname[i][3]+"entropy.npy")
	S4 = np.load(mydir[i][4]+"/"+fname[i][4]+"entropy.npy")

	plt.plot(T,S0,label="w = %g"%(w[0]),color="b")
	plt.plot(T,S1,label="w = %g"%(w[1]),color="k")
	plt.plot(T,S2, label="w = %g"%(w[2]),color="c")
	plt.plot(T,S3, label="w = %g"%(w[3]),color="g")
	plt.plot(T,S4, label="w = %g"%(w[4]),color="r")
	plt.legend(loc=5,prop={'size': 10})
	plt.xlabel('Time')
	plt.ylabel("entropy")
	plt.title("Entropy dynamics for electric field A*cos(wt) N = %d sites"%(NUM[i]))
	plt.savefig("Logs/entropy_N_%d"%(NUM[i]))
	
	plt.show()
	