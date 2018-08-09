from __future__ import print_function, absolute_import
import matplotlib.pyplot as plt
import scipy as sp
import os
import math
N = 64
bond_dim = 32
J = 0.
h = 1
tol_im = 1e-10
step_im = 0.0009765625	 
realstep =  0.000244141
tf = 1.5
h_quench = 0
J_quench = 1
dirname = "/2018-08-09_18-14-11"

os.chdir("Logs/N_%g_D_%g"%(N,bond_dim)+dirname)
fname = "t_ising_N%d_D%d_J%g_h%g_s%g_dtau%g_dt%g_tf%g_Jquench%g_hquench_%g" % \
			(N, bond_dim,J, h, tol_im, step_im,realstep,tf,J_quench,h_quench)

T = sp.load(fname+"_time.npy")
rate = sp.load(fname+"_rate.npy")
energy = sp.load(fname+"_energy.npy")
p = sp.load(fname+"_prob.npy")
sv = sp.load(fname+"_sv_sq.npy")

fig1 = plt.figure(1)
H_t = fig1.add_subplot(111)
H_t.set_xlabel('time in seconds')
H_t.set_ylabel('Energy during quench')
H_t.set_title('N_%d_D_%d'%(N,bond_dim)+'Energy')


fig2 = plt.figure(2)
P_t = fig2.add_subplot(111)
P_t.set_xlabel('time in seconds')
P_t.set_ylabel('Return Probability during quench')
P_t.set_title('N_%d_D_%d'%(N,bond_dim) +'Return Probability')

fig3 = plt.figure(3)
R_t = fig3.add_subplot(111)
R_t.set_xlabel('time in seconds')
R_t.set_ylabel('-(1/N)log(P)')
R_t.set_title('N_%d_D_%d'%(N,bond_dim) +'Rate function for Return Probability')

H_t.axvline(x=sp.pi*0.5,color="r")
H_t.axvline(x=1.5*sp.pi,color="r")
H_t.axvline(x=2.5*sp.pi,color="r")
H_t.plot(T,energy)

P_t.axvline(x=sp.pi*0.5,color="r")
P_t.axvline(x=1.5*sp.pi,color="r")
P_t.axvline(x=2.5*sp.pi,color="r")
P_t.plot(T,p)

R_t.axvline(x=sp.pi*0.5,color="r")
R_t.axvline(x=1.5*sp.pi,color="r")
R_t.axvline(x=2.5*sp.pi,color="r")
R_t.plot(T,rate)

plt.show()
'''
fig1.savefig(fname+"_energy.png")
fig2.savefig(fname+"_prob.png")
fig3.savefig(fname+"_rate.png")
'''


