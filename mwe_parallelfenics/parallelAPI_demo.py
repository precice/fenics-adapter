"""
Script which shows various FEniCS functionality which can be used to access points, cells and DOFs in parallel.
All functionalities described here have potential use in the adapter
"""
import dolfin
from dolfin import *
from fenics import MPI

rank = MPI.rank(MPI.comm_world)

parameters["ghost_mode"] = "shared_vertex"

n_intervals = 4
print("Mesh is defined to be 1D having {} vertices in unit interval [0.0, 0.1]".format(n_intervals+1))

mesh = UnitIntervalMesh(n_intervals)
V = FunctionSpace(mesh, "P", 1)
dofmap = V.dofmap()

print("DOFs in FEniCS are the global indices of points")
first_dof, last_dof = dofmap.ownership_range()
print("Rank {} : call dofmap.ownership_range() gives first and last dof as ({}, {})".format(rank, first_dof, last_dof))

unowned = dofmap.local_to_global_unowned()
print("Rank {}: local to global unowned: {}".format(rank, unowned))

sharednodes_map = dofmap.shared_nodes()
print("Rank {}: shared nodes map: {}".format(rank, sharednodes_map))

local_indices = []
global_indices = []
x = []
for vertex in vertices(mesh):
    local_indices.append(vertex.index())
    global_indices.append(dofmap.local_to_global_index(vertex.index()))
    x.append(vertex.x(0))

print("Rank {}: Local vertex indices are: {}".format(rank, local_indices))
print("Rank {}: Global vertex indices are: {}".format(rank, global_indices))
print("Rank {}: Vertex coordinates are: {}".format(rank, x))

v2d = vertex_to_dof_map(V)
d2v = dof_to_vertex_map(V)
print("Rank {}: vertex to dof map = {}".format(rank, v2d))
print("Rank {}: dof to vertex map = {}".format(rank, d2v))

function = project(Expression("x[0]", degree=1), V)
print("Rank {}: Projecting function f = x[0] (x-coordinate value) on function space".format(rank))

vec = function.vector().get_local()

global_indices = []
l_indices = []
for i in range(len(vec)):
    Vertex = dolfin.MeshEntity(mesh, 0, i)
    global_indices.append(Vertex.global_index())
    l_indices.append(Vertex.index())

print("Rank {}: global indices of local vertices are = {}".format(rank, global_indices))
print("Rank {}: local_indices of local vertices are = {}".format(rank, l_indices))
