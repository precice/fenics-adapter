"""
Minimum working example of problem
"""
from fenics import Expression, UnitSquareMesh, FunctionSpace, interpolate, Function
from fenics import MPI
from adapter import Adapter

mesh = UnitSquareMesh(10, 10)

expr = Expression("x[0] + x[1]", degree=1)
V = FunctionSpace(mesh, "P", 1)
function = interpolate(expr, V)

function2 = Function(V)
function2.rename("Function", "")

adapter = Adapter()

# adapter.set_func(function)
# print("{rank} of {size}:Setting function in Adapter".format(rank=MPI.rank(MPI.comm_world), size=MPI.size(MPI.comm_world)))
#
# updated_func = adapter.get_func()
# print("{rank} of {size}:Getting function from Adapter".format(rank=MPI.rank(MPI.comm_world), size=MPI.size(MPI.comm_world)))

adapter.set_func(function2)
print("{rank} of {size}:Setting function2 in Adapter".format(rank=MPI.rank(MPI.comm_world), size=MPI.size(MPI.comm_world)))

updated_func = adapter.get_func()
print("{rank} of {size}:Getting function2 from Adapter".format(rank=MPI.rank(MPI.comm_world), size=MPI.size(MPI.comm_world)))