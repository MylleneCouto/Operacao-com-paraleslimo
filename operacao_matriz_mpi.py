from mpi4py import MPI
import numpy as np
import time
import sys

N = 10000
M = 10000
REPEAT = 10  # Número de repetições para tornar mais pesado

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Tratamento simples: aborta se N não divisível por size
if N % size != 0:
    if rank == 0:
        print("ERRO: Número de linhas (N) não é divisível pelo número de processos.")
    sys.exit(1)

local_rows = N // size

if rank == 0:
    A = np.random.rand(N, M)
    B = np.random.rand(N, M)
else:
    A = None
    B = None

local_A = np.empty((local_rows, M), dtype='d')
local_B = np.empty((local_rows, M), dtype='d')

# SCATTER simples (sem balanceamento)
comm.Scatter([A, local_rows * M, MPI.DOUBLE], local_A, root=0)
comm.Scatter([B, local_rows * M, MPI.DOUBLE], local_B, root=0)

comm.Barrier()
t0 = time.time()

# OPERAÇÃO CUSTOSA REPETIDA
local_C = np.zeros((local_rows, M), dtype='d')
tmp = local_A.copy()

local_C = np.log1p(np.abs(np.sin(local_A) * np.cos(local_B)) + np.sqrt(local_A**2 + local_B**2 + 1.0)) ** 2.5

local_sum = np.sum(local_C)
total_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)

# GATHER
if rank == 0:
    C = np.empty((N, M), dtype='d')
else:
    C = None
comm.Gather(local_C, C, root=0)

comm.Barrier()
t1 = time.time()
elapsed = t1 - t0

if rank == 0:
    print(f"\nProcessos: {size}")
    print(f"Tempo (mais pesado): {elapsed:.4f} segundos")
    print(f"Soma total da matriz final: {total_sum:.4e}")
