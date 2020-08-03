"""
Script which shows various FEniCS functionality which can be used to access points, cells and DOFs in parallel.
All functionalities described here have potential use in the adapter

Important documentation for DofMap in dolfin: https://fenicsproject.org/docs/dolfin/1.6.0/python/programmers-reference/cpp/fem/GenericDofMap.html
"""
import dolfin
from dolfin import *
import numpy as np
from mpi4py import MPI


def determine_shared_nodes(function_space):
    # Identify owned global indices
    dofmap = function_space.dofmap()
    sharednodes_map = dofmap.shared_nodes()
    shared_ids = list(sharednodes_map.keys())
    unowned_ids = dofmap.local_to_global_unowned()
    global_ids = dofmap.tabulate_local_to_global_dofs()

    # Identify local ids of vertices which are not owned by this rank
    unowned_shared_ids, owned_shared_ids = [], []
    for sid in shared_ids:
        for unid in unowned_ids:
            if global_ids[sid] == unid:
                unowned_shared_ids.append(sid)

    unowned_ids = np.array(unowned_shared_ids)

    owned_ids = np.copy(shared_ids)
    for unowned_id in unowned_ids:
        owned_ids = owned_ids[owned_ids != unowned_id]

    return owned_ids, unowned_ids


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
# n_intervals = 7
# if rank == 0:
#     print("1D Mesh consisting of {} vertices in unit interval [0.0, 1.0]".format(n_intervals + 1))
# mesh = UnitIntervalMesh(n_intervals)

p0 = Point(0.0, 0.0)
p1 = Point(1.0, 1.0)
mesh = RectangleMesh(p0, p1, 2, 2)

V = FunctionSpace(mesh, "P", 1)
dofmap = V.dofmap()

if rank == 0:
    print("----------- Extracting entities using only function space and DoFmap ----------")

print("Rank {} shares nodes with rank {}".format(rank, dofmap.neighbours()))
comm.Barrier()

first_dof, last_dof = dofmap.ownership_range()
print("Rank {} : call dofmap.ownership_range() gives first and last dof as ({}, {})".format(rank, first_dof, last_dof))
comm.Barrier()

dof_coords = V.tabulate_dof_coordinates()
print("Rank {} : dof coordinates = {}".format(rank, dof_coords))
comm.Barrier()

print("Rank {}: shared nodes map: {}".format(rank, dofmap.shared_nodes()))
comm.Barrier()

print("Rank {}: local_to_global_unowned: {}".format(rank, dofmap.local_to_global_unowned()))
comm.Barrier()

print("Rank {}: tabulate_local_to_global_dofs: {}".format(rank, dofmap.tabulate_local_to_global_dofs()))
comm.Barrier()

if rank == 0:
    print()
    print("----------- Extracting entities using Mesh ----------")

global_indices = []
x = []
for vertex in vertices(mesh):
    global_indices.append(dofmap.local_to_global_index(vertex.index()))
    x.append(vertex.x(0))

print("Rank {}: Global vertex indices are: {}".format(rank, global_indices))
comm.Barrier()

print("Rank {}: Vertex coordinates are: {}".format(rank, x))
comm.Barrier()

if rank == 0:
    print()
    print("---------- Communicating shared node values ----------")

n_vertices = len(dofmap.tabulate_local_to_global_dofs())
owned_shared, unowned_shared = determine_shared_nodes(V)
if owned_shared.size == 0:
    print("Rank {}: This process does not own any nodes it shares".format(rank))
else:
    print("Rank {}: Indices of nodes which this process owns and shares are {}".format(rank, owned_shared))
comm.Barrier()

if unowned_shared.size == 0:
    print("Rank {}: This process does not have any shared nodes which it does not own".format(rank))
else:
    print("Rank {}: Indices of nodes which this process does not own but share are {}".format(rank, unowned_shared))
comm.Barrier()

data = np.full(n_vertices, rank)
print("Rank {}: Data before communication: {}".format(rank, data))
comm.Barrier()

sharednodes_map = dofmap.shared_nodes()
if rank % 2 == 0:
    if owned_shared.size != 0:
        for point in owned_shared:
            dest_ranks = sharednodes_map[point]
            for proc in dest_ranks:
                comm.send(data[point], dest=proc, tag=11)
    else:
        print("Rank {}: Nothing to Send".format(rank))
else:
    if unowned_shared.size != 0:
        for point in unowned_shared:
            sources = sharednodes_map[point]
            for proc in sources:
                data[point] = comm.recv(source=proc, tag=11)
    else:
        print("Rank {}: Nothing to Receive".format(rank))

if rank % 2 != 0:
    if owned_shared.size != 0:
        for point in owned_shared:
            dest_ranks = sharednodes_map[point]
            for proc in dest_ranks:
                comm.send(data[point], dest=proc, tag=11)
    else:
        print("Rank {}: Nothing to Send".format(rank))
else:
    if unowned_shared.size != 0:
        for point in unowned_shared:
            sources = sharednodes_map[point]
            for proc in sources:
                data[point] = comm.recv(source=proc, tag=11)
    else:
        print("Rank {}: Nothing to Receive".format(rank))

print("Rank {}: Data after communication: {}".format(rank, data))

function = project(Expression("x[0]", degree=1), V)
if rank == 0:
    print()
    print("---------- Introducing function f = x ----------")
    print("Projecting function f = x[0] (x-coordinate value) on function space".format(rank))

comm.Barrier()

vec = function.vector().get_local()
global_indices = []
l_indices = []
for i in range(len(vec)):
    Vertex = dolfin.MeshEntity(mesh, 0, i)
    global_indices.append(Vertex.global_index())

print("Rank {}: global indices of local vertices are = {}".format(rank, global_indices))
comm.Barrier()


