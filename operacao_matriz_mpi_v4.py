from mpi4py import MPI
import numpy as np
import time

# Em vez de scatter, broadcast a matriz inteira para todos os processos

N = 10000
M = 10000

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    A = np.random.rand(N, M)
    B = np.random.rand(N, M)
else:
    A = None
    B = None

if A is None:
    A = np.empty((N, M), dtype='d')
if B is None:
    B = np.empty((N, M), dtype='d')

comm.Bcast(A, root=0)
comm.Bcast(B, root=0)

comm.Barrier()
t0 = time.time()

# Cada processo opera sobre a matriz inteira
local_C = np.log1p(np.abs(np.sin(A) * np.cos(B)) + np.sqrt(A**2 + B**2 + 1.0)) ** 2.5

local_sum = np.sum(local_C)

total_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)

comm.Barrier()
t1 = time.time()
elapsed = t1 - t0

if rank == 0:
    print(f"\nProcessos: {size}")
    print(f"Tempo: {elapsed:.4f} segundos")
    print(f"Soma total da matriz: {total_sum:.4e}")
