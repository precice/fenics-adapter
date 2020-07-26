"""
Script which shows various FEniCS functionality which can be used to access points, cells and DOFs in parallel.
All functionalities described here have potential use in the adapter

Important documentation for DofMap in dolfin: https://fenicsproject.org/docs/dolfin/1.6.0/python/programmers-reference/cpp/fem/GenericDofMap.html
"""
import dolfin
from dolfin import *
from fenics import MPI

rank = MPI.rank(MPI.comm_world)

# parameters["ghost_mode"] = "shared_vertex"

# n_intervals = 7
# if rank == 0:
#     print("1D Mesh consisting of {} vertices in unit interval [0.0, 1.0]".format(n_intervals + 1))
# mesh = UnitIntervalMesh(n_intervals)

p0 = Point(0.0, 0.0)
p1 = Point(1.0, 1.0)
mesh = RectangleMesh(p0, p1, 2, 2)

V = FunctionSpace(mesh, "P", 1)

dofmap = V.dofmap()

print("Rank {} shares nodes with rank {}".format(rank, dofmap.neighbours()))
MPI.barrier(MPI.comm_world)

first_dof, last_dof = dofmap.ownership_range()
print("Rank {} : call dofmap.ownership_range() gives first and last dof as ({}, {})".format(rank, first_dof, last_dof))
MPI.barrier(MPI.comm_world)

dof_coords = V.tabulate_dof_coordinates()
print("Rank {} : dof coordinates = {}".format(rank, dof_coords))
MPI.barrier(MPI.comm_world)

print("Rank {}: shared nodes map: {}".format(rank, dofmap.shared_nodes()))
MPI.barrier(MPI.comm_world)

print("Rank {}: local_to_global_unowned: {}".format(rank, dofmap.local_to_global_unowned()))
MPI.barrier(MPI.comm_world)

global_indices = []
x = []
for vertex in vertices(mesh):
    global_indices.append(dofmap.local_to_global_index(vertex.index()))
    x.append(vertex.x(0))

print("Rank {}: Global vertex indices are: {}".format(rank, global_indices))
MPI.barrier(MPI.comm_world)

print("Rank {}: Vertex coordinates are: {}".format(rank, x))
MPI.barrier(MPI.comm_world)

if rank == 0:
    print("Vertex coordinates accessed by vertices(mesh) show duplicate vertex whereas DOF coordinates only show")

v2d = vertex_to_dof_map(V)
d2v = dof_to_vertex_map(V)
print("Rank {}: vertex to dof map = {}".format(rank, v2d))
MPI.barrier(MPI.comm_world)

print("Rank {}: dof to vertex map = {}".format(rank, d2v))
MPI.barrier(MPI.comm_world)

function = project(Expression("x[0]", degree=1), V)
if rank == 0:
    print("---------- Introducing function f = x ----------")
    print("Projecting function f = x[0] (x-coordinate value) on function space".format(rank))

MPI.barrier(MPI.comm_world)

vec = function.vector().get_local()
global_indices = []
l_indices = []
for i in range(len(vec)):
    Vertex = dolfin.MeshEntity(mesh, 0, i)
    global_indices.append(Vertex.global_index())

print("Rank {}: global indices of local vertices are = {}".format(rank, global_indices))
MPI.barrier(MPI.comm_world)

