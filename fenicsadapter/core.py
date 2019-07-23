"""This module handles CustomExpression and low-level functionality of the FEniCS adapter.

:raise ImportError: if PRECICE_ROOT is not defined
"""

import dolfin
from scipy.interpolate import Rbf
from scipy.interpolate import interp1d
import numpy as np


try:
    import precice
    from precice import action_read_iteration_checkpoint
    from precice import action_write_iteration_checkpoint
    from precice import action_write_initial_data
except ImportError:
    import os
    import sys
    # check if PRECICE_ROOT is defined
    if not os.getenv('PRECICE_ROOT'):
       raise Exception("ERROR: PRECICE_ROOT not defined!")

    precice_root = os.getenv('PRECICE_ROOT')
    precice_python_adapter_root = precice_root+"/src/precice/bindings/python"
    sys.path.insert(0, precice_python_adapter_root)
    import precice
    from precice import action_read_iteration_checkpoint
    from precice import action_write_iteration_checkpoint
    from precice import action_write_initial_data


def extract_subdomain_vertices(mesh, subdomain, dimension):
    """Extracts verticies which lay on the boundary. Currently handles 2D
    case properly, 3D is circumvented.

    :raise Exception: if no correct coupling interface is defined
    :return: stack of verticies
    """
    n = 0
    vertices_x = []
    vertices_y = []
    if dimension == 3:
        vertices_z = []

    if not issubclass(type(subdomain), dolfin.SubDomain):
        raise Exception("no correct coupling interface defined!")

    for v in dolfin.vertices(mesh):
        if subdomain.inside(v.point(), True):
            n += 1
            vertices_x.append(v.x(0))
            vertices_y.append(v.x(1))
            if dimension == 3:
                # todo this has to be fixed for "proper" 3D coupling. Currently this is a workaround for the coupling of 2D fenics with pseudo 3D openfoam
                vertices_z.append(0)

    if dimension == 2:
        return np.stack([vertices_x, vertices_y]), n
    elif dimension == 3:
        return np.stack([vertices_x, vertices_y, vertices_z]), n


def extract_subdomain_coordinates(mesh, subdomain, dimension):
    """Extracts the coordinates of vertices that lay on the boundary. 3D
    case currently handled as 2D.

    :return: x and y coordinates.
    """
    vertices, _ = extract_subdomain_vertices(mesh, subdomain, dimension)
    vertices_x = vertices[0, :]
    vertices_y = vertices[1, :]
    if dimension == 3:
        vertices_z = vertices[2, :]

    if dimension == 2:
        return vertices_x, vertices_y
    elif dimension == 3:
        # todo this has to be fixed for "proper" 3D coupling. Currently this is a workaround for the coupling of 2D fenics with pseudo 3D openfoam
        return vertices_x, vertices_y


def convert_fenics_to_precice(data, mesh, subdomain, dimension):
    """Converts FEniCS data of type dolfin.Function into
    Numpy array for all x and y coordinates on the boundary.

    :param data: FEniCS boundary function
    :raise Exception: if type of data cannot be handled
    :return: array of FEniCS function values at each point on the boundary
    """
    if type(data) is dolfin.Function:
        x_all, y_all = extract_subdomain_coordinates(mesh, subdomain, dimension)
        return np.array([data(x, y) for x, y in zip(x_all, y_all)])
    else:
        raise Exception("Cannot handle data type %s" % type(data))


class CustomExpression(dolfin.UserExpression):
    """Creates functional representation (for FEniCS) of nodal data
    provided by preCICE, using RBF interpolation.
    """
    def set_boundary_data(self, vals, coords_x, coords_y=None, coords_z=None):
        self._update_boundary_data(vals, coords_x, coords_y, coords_z)

    def _update_boundary_data(self, vals, coords_x, coords_y=None, coords_z=None):
        self._coords_x = coords_x
        if coords_y is None:
            coords_y = np.zeros(self._coords_x.shape)
        self._coords_y = coords_y
        if coords_z is None:
            coords_z = np.zeros(self._coords_x.shape)
        self._coords_z = coords_z

        self._vals = vals.flatten()
        assert (self._vals.shape == self._coords_x.shape)

    def update(self, other_expression):
        self._update_boundary_data(other_expression._vals, other_expression._coords_x, other_expression._coords_y, other_expression._coords_z)

    def rbf_interpol(self, x):
        if x.__len__() == 1:
            f = Rbf(self._coords_x, self._vals.flatten())
            return f(x)
        if x.__len__() == 2:
            f = Rbf(self._coords_x, self._coords_y, self._vals.flatten())
            return f(x[0], x[1])
        if x.__len__() == 3:
            f = Rbf(self._coords_x, self._coords_y, self._coords_z, self._vals.flatten())
            return f(x[0], x[1], x[2])

    def lin_interpol(self, x):
        f = interp1d(self._coords_y, self._vals, bounds_error=False, fill_value="extrapolate", kind="cubic")
        return f(x[1])

    def eval(self, value, x):
        value[0] = self.lin_interpol(x)


class Adapter:

    def __init__(self, solver_name, rank, size):

        self._interface = precice.Interface(solver_name, rank, size)

        ## coupling mesh related quantities
        self._coupling_mesh_vertices = None
        self._vertex_ids = None
        self._n_vertices = None

        # read/write buffers
        self._write_data = None
        self._read_data = None

    def configure(self, precice_config_file_name):
        self._interface.configure(precice_config_file_name)

    def read_block_scalar_data(self, read_data_name, mesh_name):
        _, n_vertices = extract_subdomain_vertices(self._fenics_mesh, self._coupling_subdomain, self._interface.get_dimensions())
        mesh_id = self._interface.get_mesh_id(mesh_name)
        read_data_id = self._interface.get_data_id(read_data_name, mesh_id)
        read_data_buffer = np.empty(n_vertices)
        self._interface.read_block_scalar_data(read_data_id, n_vertices, self._vertex_ids, read_data_buffer)
        return self._create_coupling_boundary_condition(read_data_buffer)

    def write_block_scalar_data(self, write_data_name, mesh_name, u):
        mesh_id = self._interface.get_mesh_id(mesh_name)
        write_data_id = self._interface.get_data_id(write_data_name, mesh_id)

        # sample write data at interface
        write_data = convert_fenics_to_precice(u, self._fenics_mesh, self._coupling_subdomain, self._interface.get_dimensions())

        # communication
        self._interface.write_block_scalar_data(write_data_id, self._n_vertices, self._vertex_ids, write_data)

    def initialize(self):
        return self._interface.initialize()

    def initialize_data(self):
        self._interface.initialize_data()

    def is_read_data_available(self):
        return self._interface.is_read_data_available()

    def set_coupling_mesh(self, mesh_name, fenics_mesh, coupling_subdomain):
        """Sets the coupling mesh. Called by initalize() function at the
        beginning of the simulation.
        """
        self._fenics_mesh = fenics_mesh
        self._coupling_subdomain = coupling_subdomain
        print(extract_subdomain_vertices(self._fenics_mesh, self._coupling_subdomain, self._interface.get_dimensions()))
        coupling_mesh_vertices, self._n_vertices = extract_subdomain_vertices(self._fenics_mesh, self._coupling_subdomain, self._interface.get_dimensions())
        mesh_id = self._interface.get_mesh_id(mesh_name)
        self._vertex_ids = np.empty(self._n_vertices)
        self._interface.set_mesh_vertices(mesh_id, self._n_vertices, coupling_mesh_vertices.flatten('F'), self._vertex_ids)

    def advance(self, dt):
        return self._interface.advance(dt)

    def create_coupling_boundary_condition(self, u_init):
        data = convert_fenics_to_precice(u_init, self._fenics_mesh, self._coupling_subdomain, self._interface.get_dimensions())
        return self._create_coupling_boundary_condition(data)

    def _create_coupling_boundary_condition(self, data):
        """Creates the coupling boundary conditions using CustomExpression."""
        x_vert, y_vert = extract_subdomain_coordinates(self._fenics_mesh, self._coupling_subdomain, self._interface.get_dimensions())

        try:  # works with dolfin 1.6.0
            coupling_bc_expression = CustomExpression()
        except (TypeError, KeyError):  # works with dolfin 2017.2.0
            coupling_bc_expression = CustomExpression(degree=0)
        coupling_bc_expression.set_boundary_data(data, x_vert, y_vert)
        return coupling_bc_expression

    def get_coupling_bc_expression(self):
        return self._coupling_bc_expression

    def is_action_required(self, action):
        return self._interface.is_action_required(action)

    def fulfilled_action(self, action):
        self._interface.fulfilled_action(action)

    def is_coupling_ongoing(self):
        return self._interface.is_coupling_ongoing()

    def finalize(self):
        self._interface.finalize()
