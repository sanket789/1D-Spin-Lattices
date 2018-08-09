#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import scipy as sp
import evoMPS.tdvp_gen as tdvp
import matplotlib.pyplot as plt
import math
'''
	This is function is for calculating ground state for transeverse Ising model
	Input: Instance of evoMPS_generic class , file name to save in Logs directory
	saves ground state
'''

def t_ising_ground_calc(s,grnd_fname, tol_im, step,auto_truncate=False):
	
	print("Bond dimensions: " + str(s.D))
	print()
	col_heads = ["Step", "t", "<H>", "d<H>", "eta"] #These last three are for testing the midpoint method.
	print("\t".join(col_heads))
	print()
	t = 0
	T = []
	H = []
	
	i = 0
	eta = 1
	while True:
		T.append(t)	#Time vector

		s.update(auto_truncate=auto_truncate)

		H.append(s.H_expect.real)

		row = [str(i)]
		row.append(str(t))
		row.append("%.15g" % H[-1])
		if len(H) > 1:
			dH = H[-1] - H[-2]
		else:
			dH = 0

		row.append("%.2e" % (dH.real))

		s.take_step(step, calc_Y_2s=True)
		t += 1.j * step
		eta = s.eta.real.sum()        
		row.append("%.6g" % eta)
		
		if math.fmod(i,10) == 0:
			print("\t".join(row))

		i += 1
		if eta < tol_im:
			s.save_state("DQPT/Logs/"+grnd_fname)
			print ("Ground state saved")
			break
