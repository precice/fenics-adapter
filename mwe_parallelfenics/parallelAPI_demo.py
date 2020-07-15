"""
Script which shows various FEniCS functionality which can be used to access points, cells and DOFs in parallel.
All functionalities described here have potential use in the adapter

Important documentation for DofMap in dolfin: https://fenicsproject.org/docs/dolfin/1.6.0/python/programmers-reference/cpp/fem/GenericDofMap.html
"""
import dolfin
from dolfin import *
from fenics import MPI

rank = MPI.rank(MPI.comm_world)

parameters["ghost_mode"] = "shared_vertex"

n_intervals = 4

if rank == 0:
    print("1D Mesh consisting of {} vertices in unit interval [0.0, 1.0]".format(n_intervals + 1))

mesh = UnitIntervalMesh(n_intervals)
V = FunctionSpace(mesh, "P", 1)
dofmap = V.dofmap()

first_dof, last_dof = dofmap.ownership_range()
print("Rank {} : call dofmap.ownership_range() gives first and last dof as ({}, {})".format(rank, first_dof, last_dof))
if rank == 0:
    print("It is unclear why for the combined dof ownership range we see 0, 1, 2, 3, 4, 5 (6 DOFs). The DOF No. 5 is unexplained")

MPI.barrier(MPI.comm_world)

dof_coords = V.tabulate_dof_coordinates()
print("Rank {} : dof coordinates = {}".format(rank, dof_coords))
if rank == 0:
    print("Corresponding to the ownership_range, the coordinates appear to show only 5 points: 0.0, 0.25, 0.5, 0.75, 1.0")

MPI.barrier(MPI.comm_world)

unowned = dofmap.local_to_global_unowned()
print("Rank {}: local to global unowned: {}".format(rank, unowned))
if rank == 0:
    print("Unclear what local to global unowned means")

MPI.barrier(MPI.comm_world)

sharednodes_map = dofmap.shared_nodes()
print("Rank {}: shared nodes map: {}".format(rank, sharednodes_map))

MPI.barrier(MPI.comm_world)

local_indices = []
global_indices = []
x = []
for vertex in vertices(mesh):
    local_indices.append(vertex.index())
    global_indices.append(dofmap.local_to_global_index(vertex.index()))
    x.append(vertex.x(0))

print("Rank {}: Local vertex indices are: {}".format(rank, local_indices))
MPI.barrier(MPI.comm_world)

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
    l_indices.append(Vertex.index())

print("Rank {}: global indices of local vertices are = {}".format(rank, global_indices))
MPI.barrier(MPI.comm_world)

print("Rank {}: local_indices of local vertices are = {}".format(rank, l_indices))
