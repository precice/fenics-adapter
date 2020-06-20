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
            ordered_coords = func_space.tabulate_dof_coordinates()
            total_pts, _ = ordered_coords.shape
            print("Process {}: ordered_coords = {}".format(MPI.rank(MPI.comm_world), ordered_coords))

            for i in range(total_pts):
                for n in range(n_vertices):
                    if ordered_coords[i, 0] == points[n, 0] and ordered_coords[i, 1] == points[n, 1]:
                        coupling_indices.append(i)

            coupling_indices = np.array(coupling_indices)
            print("Process {}: coupling_indices = {}".format(MPI.rank(MPI.comm_world), coupling_indices))

            if type(function) is dolfin.Function:
                func_vals = function.vector().get_local()
                vals = []
                for n in range(n_vertices):
                    vals.append(func_vals[coupling_indices[n]])
        else:
            print("Process {}: No function evaluation done as the rank has no boundary points".format(MPI.rank(MPI.comm_world)))

