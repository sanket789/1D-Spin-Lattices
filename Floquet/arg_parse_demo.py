#!/usr/bin/env python3

import numpy as np
from mpi4py import MPI
import random


def m_main(c):
	rank = c.Get_rank()
	size = c.Get_size()

	A = np.arange(rank*size,(rank+1.0)*size)


	ASQ = A*2
	ACU = A*3
	A2D = np.vstack((A,A))
	print('for rank = ',rank,' AA = ', A2D)
	'''
	gather data to rank = 0
	syntax for Gather:
	comm.Gather(sendbuf, recvbuf, root=0) 
	sendbuf = data to be gathered
	recvbuf = receiving variable for collection
	'''
	recvSQ = None
	recvCU = None
	recv2D = None
	if rank == 0:
		recvSQ = np.empty([size,size],dtype='float')
		recvCU = np.empty([size,size],dtype='float')
		recv2D = np.empty([size,size,size],dtype='float')
	

	comm.Gather(ASQ,recvSQ,root=0)
	comm.Gather(ACU,recvCU,root=0)
	comm.Gather(A2D,recv2D,root=0)

	
	if rank == 0:
		print("Gathered arraySQ: {}".format(recvSQ))
		print("Gathered arrayCU: {}".format(recvCU))
		print("Gathered array2D: {}".format(recv2D))
		np.savetxt('square.csv',recvSQ,delimiter=',')
		np.savetxt('cube.csv',recvCU,delimiter=',')
		#np.savetxt('2D.csv',recv2D,delimiter=',')


if __name__ == '__main__':
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	size = comm.Get_size()
	root = 0

	m_main(comm)
	






# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()
# root = 0

# local_array = np.random.rand(2,3)
# print("rank: {}, local_array: {}".format(rank, local_array))

# sendbuf = np.array(local_array)

# # Collect local array sizes using the high-level mpi4py gather
# #sendcounts = np.array(comm.gather(len(sendbuf), root))
# sendcounts = (2*size,3)
# if rank == root:
#     print("sendcounts: {}, total: {}".format(sendcounts, 2*size*3))
#     recvbuf = np.empty(sendcounts, dtype=float)

# else:
#     recvbuf = None

# comm.Gatherv(sendbuf=sendbuf, recvbuf=(recvbuf, 5), root=root)
# if rank == root:
#     print("Gathered array: {}".format(recvbuf))