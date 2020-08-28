"""
Script which shows various FEniCS functionality which can be used to access points, cells and DOFs in parallel.
All functionalities described here have potential use in the adapter

Important documentation for DofMap in dolfin: https://fenicsproject.org/docs/dolfin/1.6.0/python/programmers-reference/cpp/fem/GenericDofMap.html
"""
import dolfin
from dolfin import *
import numpy as np
from mpi4py import MPI
import hashlib


def determine_shared_vertices(function_space):
    dofmap = function_space.dofmap()
    sharednodes_map = dofmap.shared_nodes()

    # Global IDs of vertices seen by this rank
    global_ids = dofmap.tabulate_local_to_global_dofs()

    # Local ids of vertices which are shared by this rank
    shared_ids = list(sharednodes_map.keys())

    # Global IDs of vertices which are not owned by this rank
    unowned_ids = dofmap.local_to_global_unowned()

    # Identify local IDs of vertices which are not owned by this rank
    unowned_shared_ids, owned_shared_ids = [], []
    for sid in shared_ids:
        for unid in unowned_ids:
            if global_ids[sid] == unid:
                unowned_shared_ids.append(sid)
    unowned_ids = np.array(unowned_shared_ids)

    # Identify local IDs of vertices which are owned by this rank
    owned_ids = np.copy(shared_ids)
    for unowned_id in unowned_ids:
        owned_ids = owned_ids[owned_ids != unowned_id]

    return owned_ids, unowned_ids


def communicate_shared_vertices(dofmap, data, owned_ids, unowned_ids):
    sharednodes_map = dofmap.shared_nodes()

    hash_tag = hashlib.sha256()

    if unowned_ids.size != 0:
        for point in unowned_ids:
            for source in sharednodes_map[point]:
                hash_tag.update((str(source) + str(rank)).encode('utf-8'))
                tag = int(hash_tag.hexdigest()[:6], base=16)
                req = comm.irecv(source=source, tag=tag)
                data[point] = req.wait()
    else:
        print("Rank {}: Nothing to Receive".format(rank))

    requests = []
    if owned_ids.size != 0:
        for point in owned_ids:
            for dest in sharednodes_map[point]:
                hash_tag.update((str(rank) + str(dest)).encode('utf-8'))
                tag = int(hash_tag.hexdigest()[:6], base=16)
                req = comm.isend(data[point], dest=dest, tag=tag)
                requests.append(req)
    else:
        print("Rank {}: Nothing to Send".format(rank))

    MPI.Request.Waitall(requests)


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
    print("---------- Communicating shared node values ----------")

n_vertices = len(dofmap.tabulate_local_to_global_dofs())
owned_shared, unowned_shared = determine_shared_vertices(V)
if owned_shared.size == 0:
    print("Rank {}: This process does not own any vertices it shares".format(rank))
else:
    print("Rank {}: Indices of vertices which this process owns and shares are {}".format(rank, owned_shared))
comm.Barrier()

if unowned_shared.size == 0:
    print("Rank {}: This process owns all the vertices it shares".format(rank))
else:
    print("Rank {}: Indices of vertices which this process does not own but shares are {}".format(rank, unowned_shared))
comm.Barrier()

data = np.full(n_vertices, rank)
print("Rank {}: Data before communication: {}".format(rank, data))
comm.Barrier()

communicate_shared_vertices(dofmap, data, owned_shared, unowned_shared)

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


