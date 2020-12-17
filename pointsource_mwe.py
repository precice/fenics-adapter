"""
Minimum working example to check if defining PointSource objects in parallel leads to hanging
"""
from fenics import Point, PointSource, SubDomain, RectangleMesh, vertices, VectorFunctionSpace
import numpy as np
from mpi4py import MPI

H = 1
W = 0.1
tol = 1E-14


class neumannBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14
        if on_boundary and ((abs(x[1] - 1) < tol) or abs(abs(x[0]) - W / 2) < tol):
            return True
        else:
            return False


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# create Mesh
n_x_Direction = 1
n_y_Direction = 10
mesh = RectangleMesh(Point(-W / 2, 0), Point(W / 2, H), n_x_Direction, n_y_Direction)
boundary = neumannBoundary()

fenics_coords = []
for v in vertices(mesh):
    if boundary.inside(v.point(), True):
        fenics_coords.append([v.x(0), v.x(1)])

fenics_coords = np.array(fenics_coords)
n_vertices, _ = fenics_coords.shape

# create Function Space
V = VectorFunctionSpace(mesh, 'P', 2)

# Dummy nodal data to assign as PointSource value
nodal_data = np.zeros_like(fenics_coords)

# Define PointSource for all points on coupling boundary
vertices_x = fenics_coords[:, 0]
vertices_y = fenics_coords[:, 1]

x_forces = dict()
y_forces = dict()

for i in range(n_vertices):
    px, py = vertices_x[i], vertices_y[i]
    key = (px, py)
    print("Rank{}: Before defining PointSource for point {}".format(rank, key))
    x_forces[key] = PointSource(V.sub(0), Point(px, py), nodal_data[i, 0])
    y_forces[key] = PointSource(V.sub(1), Point(px, py), nodal_data[i, 1])
    print("Rank {}: After defining PointSource for point {}".format(rank, key))

print("Rank {}: Definition of PointSource's is successful".format(rank))
