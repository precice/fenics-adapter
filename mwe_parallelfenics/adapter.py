"""
Dummy adapter for MWE
"""
import dolfin
import numpy as np
from fenics import MPI

class Adapter:
    def __init__(self):
        self._function = None

    def set_vertices(self, mesh, subdomain):
        vertices_x = []
        vertices_y = []
        for v in dolfin.vertices(mesh):
            if subdomain.inside(v.point(), True):
                vertices_x.append(v.x(0))
                vertices_y.append(v.x(1))
        return np.stack([vertices_x, vertices_y], axis=1)

    def eval_func(self, function, points):
        if type(function) is dolfin.Function:
            n_vertices, _ = points.shape
            if n_vertices > 0:
                x_all, y_all = points[:, 0], points[:, 1]
                for x, y in zip(x_all, y_all):
                    print("(x,y) = ({},{})".format(x, y))
                    print("Process {}: function evaluation at ({},{}) = {}".format(MPI.rank(MPI.comm_world), x, y, function(x, y)))
            else:
                print("Process {} does not have any vertices".format(MPI.rank(MPI.comm_world)))

