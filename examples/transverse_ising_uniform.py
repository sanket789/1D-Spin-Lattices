#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A demonstration of evoMPS: Calculation of approximate excitation spectrum
for the transverse Ising model.

@author: Ashley Milsted
"""
#from __future__ import absolute_import, division, print_function

import copy
import scipy as sp
import scipy.linalg as la
import scipy.special as spe
import evoMPS.tdvp_uniform as tdvp
import evoMPS.matmul as matmul
"""
First, we set up some global variables to be used as parameters.
"""

bond_dim = 32                 #The maximum bond dimension

J = 1.00                      #Interaction factor
h = 0.50                     #Transverse field factor

tol_im = 1E-10                #Ground state tolerance (norm of projected evolution vector)

step = 0.08                   #Imaginary time step size

load_saved_ground = True      #Whether to load a saved ground state (if it exists)

auto_truncate = False         #Whether to reduce the bond-dimension if any Schmidt coefficients fall below a tolerance.
zero_tol = 1E-20              #Zero-tolerance for the Schmidt coefficients squared (right canonical form)

num_excitations = 24          #The number of excitations to obtain
num_momenta = 20              #Number of points on momentum axis
top_non_triv = True           #Also look for topologically non-trivial excitations (only useful for h < J)

plot_results = True

sanity_checks = False         #Whether to perform additional (verbose) sanity checks

"""
Next, we define our Hamiltonian and some observables.
"""
Sx = sp.array([[0, 1],
                 [1, 0]])
Sy = 1.j * sp.array([[0, -1],
                       [1, 0]])
Sz = sp.array([[1, 0],
                 [0, -1]])

"""
A translation invariant (uniform) nearest-neighbour Hamiltonian is a
4-dimensional array defining the nearest-neighbour interaction.
The indices 0 and 1 are the 'bra' indices for the first and
second sites and the indices 2 and 3 are the 'ket' indices:

  ham[s,t,u,v] = <st|h|uv>

The following function will return a Hamiltonian for the chain, given the
the parameters J and h.
"""
def get_ham(J, h):
    ham = -J * (sp.kron(Sx, Sx) + h * sp.kron(Sz, sp.eye(2))).reshape(2, 2, 2, 2)
    return ham

lam = J / h
print("Exact energy = ", -h * 2 / sp.pi * (1 + lam) * spe.ellipe((4 * lam / (1 + lam)**2)))
def contract_uni(A,B):
    M = sp.dot(matmul.H(A[0][0]),B[0][0]) + sp.dot(matmul.H(A[0][1]),B[0][1])
    tr = sp.trace(matmul.H(A[0][0]))*sp.trace(B[0][0]) + sp.trace(matmul.H(A[0][1]))*sp.trace(B[0][1])
    return tr.real
"""
Now we are ready to create an instance of the evoMPS class.
"""
s = tdvp.EvoMPS_TDVP_Uniform(bond_dim, 2, get_ham(J, h))
s.zero_tol = zero_tol
s.sanity_checks = sanity_checks

"""
The following loads a ground state from a file.
The ground state will be saved automatically when it is declared found.
"""
grnd_fname = "t_ising_uni_D%d_J%g_h%g_s%g_dtau%g_ground.npy" % (bond_dim, J, h, tol_im, step)

if load_saved_ground:
    try:
        a_file = open(grnd_fname, 'rb')
        s.load_state(a_file)
        a_file.close
        real_time = True
        loaded = True
        print('Using saved ground state: ' + grnd_fname)
    except IOError as e:
        real_time = False
        loaded = False
        print('No existing ground state could be opened.')
else:
    real_time = False
    loaded = False


if __name__ == '__main__':
    """
    Prepare some loop variables and some vectors to hold data from each step.
    """
    t = 0

    T = []
    H = []
    M = []

    """
    Print a table header.
    """
    print("Bond dimensions: " + str(s.D))
    print()
    col_heads = ["Step", "t", "<h>", "d<h>",
                 "sig_x", "sig_y", "sig_z",
                 "eta"] #These last three are for testing the midpoint method.
    print("\t".join(col_heads))
    print()

    """
    The following loop performs Euler integration of imaginary time evolution.
    """
    eta = 1
    i = 0
    while True:
        T.append(t)

        """
        Update secondary data used to calculate expectation values etc.
        This must also be done before calling take_step().
        """
        s.update(auto_truncate=auto_truncate) 

        H.append(s.h_expect.real)

        row = [str(i)]
        row.append(str(t))
        row.append("%.15g" % H[-1])

        if len(H) > 1:
            dH = H[-1] - H[-2]
        else:
            dH = 0

        row.append("%.2e" % (dH.real))

        """
        Compute expectation values!
        """
        exSx = s.expect_1s(Sx)
        exSy = s.expect_1s(Sy)
        exSz = s.expect_1s(Sz)
        row.append("%.3g" % exSx.real)
        row.append("%.3g" % exSy.real)
        row.append("%.3g" % exSz.real)

        M.append(exSz.real)

        """
        Carry out next step!
        """
        s.take_step(step)
        t += 1.j * step

        eta = s.eta.real
        row.append("%.6g" % eta)

        print("\t".join(row))

        i += 1

        if eta < tol_im or loaded:
            s.update()
            s.save_state(grnd_fname)
            break
    norm = contract_uni(sp.asarray(s.A),sp.asarray(s.A))
    print "norm of ground state is = ",norm
    print s.A[0].shape
    print sp.linalg.norm(s.A[0])
'''
    """
    Find excitations once we have the ground state.
    """
    print('Finding excitations!')
    if top_non_triv:
        s2 = copy.deepcopy(s)
        s2.apply_op_1s(Sz)
        s2.update()
        print("Energy density difference with spin flip:", s.h_expect.real - s2.h_expect.real)
    ex_ev = []
    ex_ev_nt = []
    ex_p = []
    for p in sp.linspace(0, sp.pi, num=num_momenta):
        print("p = ", p)
        ex_ev.append(s.excite_top_triv(p, nev=num_excitations, ncv=num_excitations * 4))
        if top_non_triv:
            ex_ev_nt.append(s.excite_top_nontriv(s2, p, nev=num_excitations, ncv=num_excitations * 4))
        else:
            ex_ev_nt.append([0])
        ex_p.append([p] * num_excitations)
    
    """
    Simple plots of the results.
    """
    if plot_results:
        import matplotlib.pyplot as plt

        if not loaded: #Plot imaginary time evolution of K1 and Mx
            tau = sp.array(T).imag

            fig1 = plt.figure(1)
            fig2 = plt.figure(2)
            H_tau = fig1.add_subplot(111)
            H_tau.set_xlabel('tau')
            H_tau.set_ylabel('H')
            H_tau.set_title('Imaginary time evolution: Energy')
            M_tau = fig2.add_subplot(111)
            M_tau.set_xlabel('tau')
            M_tau.set_ylabel('M')
            M_tau.set_title('Imaginary time evolution: Magnetization')

            H_tau.plot(tau, H)
            M_tau.plot(tau, M)

        plt.figure()
        ex_p = sp.array(ex_p).ravel()
        ex_ev = sp.array(ex_ev).ravel()
        ex_ev_nt = sp.array(ex_ev_nt).ravel()
        plt.plot(ex_p, ex_ev, 'bo', label='top. trivial')
        if top_non_triv:
            plt.plot(ex_p, ex_ev_nt, 'ro', label='top. non-trivial')

        elem_ex = lambda p: 2 * J * sp.sqrt(1 + h**2 / J**2 - 2 * h / J * sp.cos(p))
        p = sp.linspace(0, sp.pi, num=100)
        plt.plot(p, elem_ex(p), 'c-', label='exact elem. excitation')

        plt.title('Excitation spectrum')
        plt.xlabel('p')
        plt.ylabel('dE')
        plt.ylim(0, max(ex_ev.max(), ex_ev_nt.max()) * 1.1)
        plt.legend()

        plt.show()
'''