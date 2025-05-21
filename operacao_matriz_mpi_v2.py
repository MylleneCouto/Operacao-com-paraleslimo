from mpi4py import MPI
import numpy as np
import time

# Variante Balanceada:
# Ajusta a distribuição das linhas usando Scatterv e Gatherv, permitindo
# balanceamento automático mesmo quando N não é divisível pelo número de processos.

N = 10000
M = 10000

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

rows = [N // size + (1 if i < N % size else 0) for i in range(size)]
displs = [sum(rows[:i]) for i in range(size)]

local_rows = rows[rank]

if rank == 0:
    A = np.random.rand(N, M)
    B = np.random.rand(N, M)
else:
    A = None
    B = None

local_A = np.empty((local_rows, M), dtype='d')
local_B = np.empty((local_rows, M), dtype='d')

sendcounts = [r * M for r in rows]
displs_counts = [d * M for d in displs]

comm.Scatterv([A, sendcounts, displs_counts, MPI.DOUBLE], local_A, root=0)
comm.Scatterv([B, sendcounts, displs_counts, MPI.DOUBLE], local_B, root=0)

comm.Barrier()
t0 = time.time()

local_C = np.log1p(np.abs(np.sin(local_A) * np.cos(local_B)) + np.sqrt(local_A**2 + local_B**2 + 1.0)) ** 2.5

local_sum = np.sum(local_C)
total_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)

if rank == 0:
    C = np.empty((N, M), dtype='d')
else:
    C = None
comm.Gatherv(local_C, [C, sendcounts, displs_counts, MPI.DOUBLE], root=0)

comm.Barrier()
t1 = time.time()
elapsed = t1 - t0

if rank == 0:
    print(f"\nProcessos: {size}")
    print(f"Tempo (balanceado): {elapsed:.4f} segundos")
    print(f"Soma total da matriz final: {total_sum:.4e}")
