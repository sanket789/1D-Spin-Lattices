import numpy as np
import matplotlib.pyplot as plt
import os
import numpy.fft as f


NUM = 512

J = 1.0
dt = 0.1
tf = 1000.
A = 0.5
sigma = 0.5*(np.sqrt(5) + 1)	#irrational number for quasi-periodicity
alpha = 0. 		#random num

N = int(tf/dt)
T = [dt*n for n in range(N)]

w0 = np.pi
w1 = np.pi/2.
w2 = np.pi/4.
w3 = np.pi/8.
w = [w0,w1,w2,w3]

mu0 = [0.0,0.5,2.0,3.0]

mydir = [[],[],[],[]]
fname = [[],[],[],[]]
folder =[["2018-09-21_05-03-36","2018-09-21_06-33-19","2018-09-21_13-57-44","2018-09-21_07-36-05"],
		 ["2018-09-21_06-00-40","2018-09-21_07-04-21","2018-09-21_14-16-56","2018-09-21_07-36-18"],
		 ["2018-09-21_06-20-16","2018-09-21_07-04-39","2018-09-21_14-17-12","2018-09-21_07-36-31"],
		 ["2018-09-21_06-32-50","2018-09-21_07-04-52","2018-09-21_14-17-28","2018-09-21_07-36-38"]]

for i in range(4):
	for j in range(4):

		mydir[i].append(os.path.join(os.getcwd(), "Logs/","N_%g_w_%g"%(NUM,w[j]),folder[i][j]))
		fname[i].append("ALT"+"FLUX_N_%d_J_%g_w_%g_mu0_%g_s_%g_al_%g_tf_%g_dt_%g_"%(NUM,J,w[j],mu0[i],sigma,alpha,tf,dt))


for i in range(4):

	
	#plot nbar
	NB0 = np.load(mydir[i][0]+"/"+fname[i][0]+"nbar.npy")
	NB1 = np.load(mydir[i][1]+"/"+fname[i][1]+"nbar.npy")
	NB2 = np.load(mydir[i][2]+"/"+fname[i][2]+"nbar.npy")
	NB3 = np.load(mydir[i][3]+"/"+fname[i][3]+"nbar.npy")
	#NB4 = np.load(mydir[i][4]+"/"+fname[i][4]+"nbar.npy")	
	half = 0.5*np.ones(N)
	plt.plot(T,NB0.real - half,label="w = %g"%(w[0]),color="b")
	plt.plot(T,NB1.real - half,label="w = %g"%(w[1]),color="k")
	plt.plot(T,NB2.real - half, label="w = %g"%(w[2]),color="r")
	plt.plot(T,NB3.real - half, label="w = %g"%(w[3]),color="g")
	#plt.plot(T,NB4.real - half, label="w = %g"%(w[4]),color="c")
	plt.legend(loc=0,prop={'size': 10})
	plt.xlabel('Time')
	plt.ylabel("local occupation number ")
	plt.title("Local occupation number for N = %d sites mu0 = %g"%(NUM,mu0[i]),loc='right')
	plt.savefig("Logs/plots/Sep21/nbar_N_%d_mu0_%g.png"%(NUM,mu0[i]))
	
	plt.show()
	

	#plot entropy

	S0 = np.load(mydir[i][0]+"/"+fname[i][0]+"entropy.npy")
	S1 = np.load(mydir[i][1]+"/"+fname[i][1]+"entropy.npy")
	S2 = np.load(mydir[i][2]+"/"+fname[i][2]+"entropy.npy")
	S3 = np.load(mydir[i][3]+"/"+fname[i][3]+"entropy.npy")
	#S4 = np.load(mydir[i][4]+"/"+fname[i][4]+"entropy.npy")

	plt.plot(T,S0,label="w = %g"%(w[0]),color="b")
	plt.plot(T,S1,label="w = %g"%(w[1]),color="k")
	plt.plot(T,S2, label="w = %g"%(w[2]),color="r")
	plt.plot(T,S3, label="w = %g"%(w[3]),color="g")
	#plt.plot(T,S4, label="w = %g"%(w[4]),color="c")
	plt.axhline(y= np.log(2.**(10.)),color="r",linestyle='--')
	plt.legend(loc=5,prop={'size': 10})
	plt.xlabel('Time')
	plt.ylabel("entropy")
	plt.title("Entropy dynamics  N = %d sites mu0 = %g"%(NUM,mu0[i]))
	plt.savefig("Logs/plots/Sep21/entropy_N_%d_mu0_%g.png"%(NUM,mu0[i]))
	
	plt.show()
	
	
	
	j0 = np.load(mydir[i][0]+"/"+fname[i][0]+"current.npy")
	j1 = np.load(mydir[i][1]+"/"+fname[i][1]+"current.npy")
	j2 = np.load(mydir[i][2]+"/"+fname[i][2]+"current.npy")
	j3 = np.load(mydir[i][3]+"/"+fname[i][3]+"current.npy")
	
	plt.plot(T,j0[:,NUM//2].real,label="w = %g"%(w[0]),color="b")
	plt.plot(T,j1[:,NUM//2].real,label="w = %g"%(w[1]),color="k")
	plt.plot(T,j2[:,NUM//2].real, label="w = %g"%(w[2]),color="r")
	plt.plot(T,j3[:,NUM//2].real, label="w = %g"%(w[3]),color="g")
	
	plt.legend(loc=5,prop={'size': 10})
	plt.xlabel('Time')
	plt.ylabel("Current at half chain ")
	plt.title("Current at half chain vs time  mu0 = %g"%(mu0[i]))
	plt.savefig("Logs/plots/Sep21/halfCurrent_N_%d_mu0_%g.png"%(NUM,mu0[i]))
	
	plt.show()
	
	plt.plot(range(NUM),j0[-1].real,label="w = %g"%(w[0]),color="b")
	plt.plot(range(NUM),j1[-1].real,label="w = %g"%(w[1]),color="k")
	plt.plot(range(NUM),j2[-1].real, label="w = %g"%(w[2]),color="r")
	plt.plot(range(NUM),j3[-1].real, label="w = %g"%(w[3]),color="g")
	plt.legend(loc=5,prop={'size': 10})
	plt.xlabel('Site index')
	plt.ylabel("Current at t = %g "%(tf))
	plt.title("Current vs site index at t = %g mu0 = %g"%(tf,mu0[i]))
	plt.savefig("Logs/plots/Sep21/FinalCurrent_N_%d_mu0_%g.png"%(NUM,mu0[i]))
	
	plt.show()
	
	jw0 = f.fftshift(f.fft(j0[:,NUM//2]))
	jw1 = f.fftshift(f.fft(j1[:,NUM//2]))
	jw2 = f.fftshift(f.fft(j2[:,NUM//2]))
	jw3 = f.fftshift(f.fft(j3[:,NUM//2]))
	fftW = f.fftshift(f.fftfreq(N,dt))
	
	plt.plot(fftW,abs(jw0)**2,label="w = %g"%(w[0]),color="b")
	plt.plot(fftW,abs(jw1)**2,label="w = %g"%(w[1]),color="k")
	plt.plot(fftW,abs(jw2)**2, label="w = %g"%(w[2]),color="r")
	plt.plot(fftW,abs(jw3)**2, label="w = %g"%(w[3]),color="g")
	
	plt.legend(loc=5,prop={'size': 10})
	plt.xlabel('fourier frequency (w)')
	plt.ylabel("Current at half chain (Fourier modes)")
	plt.title("DFT Current at half chain mu0 = %g"%(mu0[i]))
	
	plt.savefig("Logs/plots/Sep21/FFT_half_Current_N_%d_mu0_%g.png"%(NUM,mu0[i]))
	
	plt.show()

