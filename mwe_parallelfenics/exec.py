"""
Minimum working example of problem of code hanging when evaluating a FEniCS Function inside an external adapter module
Run: mpirun -np N python3 exec.py
N = number of processes
"""
from fenics import UnitSquareMesh, FunctionSpace, Function
from fenics import MPI
from adapter import Adapter
import numpy as np

mesh = UnitSquareMesh(10, 10)
n_vertices = 11
vertices_x = [0.5 for _ in range(n_vertices)]
vertices_y = np.linspace(0, 1, n_vertices)
points = np.stack([vertices_x, vertices_y], axis=1)

V = FunctionSpace(mesh, "P", 1)

function = Function(V)
function.rename("Function", "")

adapter = Adapter()

adapter.eval_func(function, points)
print("Process {rank}:Evaluated in Adapter".format(rank=MPI.rank(MPI.comm_world)))
