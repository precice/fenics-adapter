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

    # Local IDs of vertices shared by this rank
    local_ids = sharednodes_map.keys()

    # Global IDs of vertices visible to this rank
    global_ids = dofmap.tabulate_local_to_global_dofs()

    # Global IDs of vertices shared by this rank
    global_shared_ids = []
    for id in local_ids:
        global_shared_ids.append(global_ids[id])

    # Global IDs of vertices which are not owned but shared by this rank
    unowned_shared_ids = dofmap.local_to_global_unowned()

    # Global IDs of vertices which are owned and shared by this rank
    owned_shared_ids = [i for i in global_shared_ids if i not in unowned_shared_ids]

    hash_tag = hashlib.sha256()

    recv_reqs = []
    for neigh in dofmap.neighbours():
        hash_tag.update((str(neigh) + str(rank)).encode('utf-8'))
        tag = int(hash_tag.hexdigest()[:6], base=16)
        recv_reqs.append(comm.irecv(source=neigh, tag=tag))

    send_reqs = []
    for neigh in dofmap.neighbours():
        hash_tag.update((str(rank) + str(neigh)).encode('utf-8'))
        tag = int(hash_tag.hexdigest()[:6], base=16)
        req = comm.isend(owned_shared_ids, dest=neigh, tag=tag)
        send_reqs.append(req)

    # Wait for all non-blocking communications to complete
    MPI.Request.Waitall(send_reqs)

    all_owned_data = dict()
    # Set received data into the existing data array
    counter = 0
    for neigh in dofmap.neighbours():
        all_owned_data[neigh] = recv_reqs[counter].wait()
        counter += 1

    to_recv_ids = dict()
    for neigh in dofmap.neighbours():
        for oid in all_owned_data[neigh]:
            for id in unowned_shared_ids:
                if oid == id:
                    to_recv_ids[id] = neigh

    local_to_global_ids = dofmap.tabulate_local_to_global_dofs()
    to_send_ids = dict()
    for id in owned_shared_ids:
        lid = int(np.where(local_to_global_ids == id)[0][0])
        for shared_id in sharednodes_map.keys():
            if shared_id == lid:
                to_send_ids[id] = sharednodes_map[lid]

    return to_send_ids, to_recv_ids


def communicate_shared_vertices(dofmap, data, to_send_ids, to_recv_ids):
    sharednodes_map = dofmap.shared_nodes()
    hash_tag = hashlib.sha256()
    local_to_global_ids = dofmap.tabulate_local_to_global_dofs()

    recv_reqs = []
    if to_recv_ids.size != 0:
        for recv_gid in to_recv_ids:
            recv_lid = int(np.where(local_to_global_ids == recv_gid)[0][0])
            for rk in sharednodes_map[recv_lid]:
                hash_tag.update((str(rk) + str(recv_gid) + str(rank)).encode('utf-8'))
                tag = int(hash_tag.hexdigest()[:6], base=16)
                print("Tag ({}): Rank {} receiving ID:({}) from rank {}".format(tag, rank, recv_gid, rk))
                recv_reqs.append(comm.irecv(source=rk, tag=tag))
                #print("Tag ({}): Rank {} received ID:({}) from rank {}".format(tag, rank, recv_gid, rk))
    else:
        print("Rank {}: Nothing to Receive".format(rank))

    send_reqs = []
    if to_send_ids.size != 0:
        for send_gid in to_send_ids:
            send_lid = int(np.where(local_to_global_ids == send_gid)[0][0])
            for rk in sharednodes_map[send_lid]:
                hash_tag.update((str(rank) + str(send_gid) + str(rk)).encode('utf-8'))
                tag = int(hash_tag.hexdigest()[:6], base=16)
                #print("Tag ({}): Rank {} sending ID: ({}) to rank {}".format(tag, rank, send_gid, rk))
                req = comm.isend(data[send_lid], dest=rk, tag=tag)
                print("Tag ({}): Rank {} sent ID: ({}) to rank {}".format(tag, rank, send_gid, rk))
                send_reqs.append(req)
    else:
        print("Rank {}: Nothing to Send".format(rank))

    # Wait for all non-blocking communications to complete
    MPI.Request.Waitall(send_reqs)

    # Set received data into the existing data array
    counter = 0
    for recv_gid in to_recv_ids:
        recv_lid = int(np.where(local_to_global_ids == recv_gid)[0][0])
        data[recv_lid] = recv_reqs[counter].wait()
        counter += 1


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

dof_coords = V.tabulate_dof_coordinates()
print("Rank {} : dof coordinates = {}".format(rank, dof_coords))
comm.Barrier()

print("Rank {}: shared nodes map: {}".format(rank, dofmap.shared_nodes()))
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

#data_vals = np.full(n_vertices, rank)
#data = {tuple(key): value for key, value in zip(dof_coords, data_vals)}
data = np.full(n_vertices, rank)
print("Rank {}: Data before communication: {}".format(rank, data))
comm.Barrier()

communicate_shared_vertices(dofmap, data, owned_shared, unowned_shared)

print("Rank {}: Data after communication: {}".format(rank, data))

# function = project(Expression("x[0]", degree=1), V)
# if rank == 0:
#     print()
#     print("---------- Introducing function f = x ----------")
#     print("Projecting function f = x[0] (x-coordinate value) on function space".format(rank))
#
# comm.Barrier()
#
# vec = function.vector().get_local()
# global_indices = []
# l_indices = []
# for i in range(len(vec)):
#     Vertex = dolfin.MeshEntity(mesh, 0, i)
#     global_indices.append(Vertex.global_index())
#
# print("Rank {}: global indices of local vertices are = {}".format(rank, global_indices))
# comm.Barrier()


