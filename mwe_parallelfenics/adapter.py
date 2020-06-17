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

    def eval_func(self, function, func_space, points):
        n_vertices, _ = points.shape
        if n_vertices > 0:
            coupling_indices = []
            for point in points:
                coupling_indices.append(np.where(func_space.tabulate_dof_coordinates() == point))

            if type(function) is dolfin.Function:
                func_vals = function.vector().get_local()
                for n in range(n_vertices):
                    print("function eval at ({},{}) = {}".format(points[n, 0], points[n, 1], func_vals[n]))

