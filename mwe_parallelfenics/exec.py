"""
Minimum working example of problem of code hanging when evaluating a FEniCS Function inside an external adapter module
Run: mpirun -np N python3 exec.py
N = number of processes
"""
from fenics import UnitSquareMesh, FunctionSpace, Function, SubDomain, near
from fenics import MPI
from adapter import Adapter


class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 1.0)


mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, "P", 1)
function = Function(V)
function.rename("Function", "")

adapter = Adapter()
domain = Right()


points = adapter.set_vertices(mesh, domain)
print("Process {rank}: Set boundary vertices".format(rank=MPI.rank(MPI.comm_world)))
print("({}): Points = {}".format(MPI.rank(MPI.comm_world), points))

adapter.eval_func(function, V, points)
print("Process {rank}: Function evaluation inside adapter is done".format(rank=MPI.rank(MPI.comm_world)))
