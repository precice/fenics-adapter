""" This module handles CustomExpression and initialization of the FEniCS adapter.
:raise ImportError: if PRECICE_ROOT is not defined
"""

import numpy as np
from .config import Config
import logging
import precice
from precice import action_write_initial_data, action_write_iteration_checkpoint, action_read_iteration_checkpoint
from .adapter_core import FunctionType, GeneralInterpolationExpression, ExactInterpolationExpression, determine_function_type,\
    InterpolationType, can_apply_2d_3d_coupling, convert_fenics_to_precice, extract_coupling_boundary_vertices, \
    extract_coupling_boundary_edges, extract_coupling_boundary_coordinates, get_forces_as_point_sources
from .solverstate import SolverState

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class Adapter:
    """Initializes the Adapter. Initalizer creates object of class Config (from
    config.py module).

    :ivar _config: object of class Config, which stores data from the JSON config file
    """
    def __init__(self, adapter_config_filename='precice-adapter-config.json'):

        self._config = Config(adapter_config_filename)

        self._solver_name = self._config.get_solver_name()

        self._interface = precice.Interface(self._solver_name, self._config.get_config_file_name(), 0, 1)
        self._dimensions = self._interface.get_dimensions()

        # FEniCS related quantities
        self._coupling_subdomain = None  # initialized later
        self._mesh_fenics = None  # initialized later
        self._coupling_bc_expression = None  # initialized later
        self._fenics_dimensions = None  # initialized later

        # coupling mesh related quantities
        self._coupling_mesh_vertices = None  # initialized later
        self._mesh_name = self._config.get_coupling_mesh_name()
        self._mesh_id = self._interface.get_mesh_id(self._mesh_name)
        self._vertex_ids = None  # initialized later
        self._n_vertices = None  # initialized later
        self._fenics_vertices = None  # initialized later

        # write data related quantities (write data is written by user from FEniCS to preCICE)
        self._write_data_name = self._config.get_write_data_name()
        self._write_data_id = self._interface.get_data_id(self._write_data_name, self._mesh_id)
        self._write_data = None  # a numpy 1D array with the values like it is used by precice (The 2D-format of values is (d0x, d0y, d1x, d1y, ..., dnx, dny) The 3D-format of values is (d0x, d0y, d0z, d1x, d1y, d1z, ..., dnx, dny, dnz))
        self._write_function_type = None  # stores whether write function is scalar or vector valued

        # read data related quantities (read data is read by use to FEniCS from preCICE)
        self._read_data_name = self._config.get_read_data_name()
        self._read_data_id = self._interface.get_data_id(self._read_data_name, self._mesh_id)
        self._read_data = None  # a numpy 1D array with the values like it is used by precice (The 2D-format of values is (d0x, d0y, d1x, d1y, ..., dnx, dny) The 3D-format of values is (d0x, d0y, d0z, d1x, d1y, d1z, ..., dnx, dny, dnz))
        self._read_function_type = None  # stores whether read function is scalar or vector valued

        # numerics
        self._precice_tau = None

        # Interpolation strategy as provided by the user
        self._my_expression = None  # initalized later

        # Solver state used by the Adapter internally to handle checkpointing
        self._checkpoint = None

        # function space
        self._function_space = None
        self._dss = None  # measure for boundary integral

        # Nodes with Dirichlet and Force-boundary
        self._Dirichlet_Boundary = None  # stores a dirichlet boundary (if provided)
        self._has_force_boundary = None  # stores whether force_boundary exists

    def set_interpolation_type(self, interpolation_type):
        """
        Sets interpolation strategy according to choice of user
        :param interpolation_type: Enum stating which interpolation strategy to be used
        (Choices are 1. CUBIC_SPLINE  2. RBF)
        """
        if interpolation_type == InterpolationType.CUBIC_SPLINE:
            self._my_expression = ExactInterpolationExpression
            print("Using cubic spline interpolation")
        elif interpolation_type == InterpolationType.RBF:
            self._my_expression = GeneralInterpolationExpression
            print("Using RBF interpolation")

    def read(self):
        """ Reads data from preCICE. Depending on the dimensions of the simulation (2D-3D Coupling, 2D-2D coupling or
        Scalar/Vector write function) read_data is converted.
        Note: For quasi 2D fenics in a 3D coupled simulation the y component of the vectors is deleted.

        :return: data read from preCICE in the form of a numpy 1D array with the values like it is used by preCICE
        (The 2D-format of values is (d0x, d0y, d1x, d1y, ..., dnx, dny)
        The 3D-format of values is (d0x, d0y, d0z, d1x, d1y, d1z, ..., dnx, dny, dnz))
        """

        assert (self._read_function_type in list(FunctionType))

        if self._read_function_type is FunctionType.SCALAR:
            self._read_data = self._interface.read_block_scalar_data(self._read_data_id, self._vertex_ids)

        elif self._read_function_type is FunctionType.VECTOR:
            if self._fenics_dimensions == self._dimensions:
                self._read_data = self._interface.read_block_vector_data(self._read_data_id, self._vertex_ids)

            elif can_apply_2d_3d_coupling(self._fenics_dimensions, self._dimensions):
                precice_read_data = self._interface.read_block_vector_data(self._read_data_id, self._vertex_ids)

                self._read_data[:, 0] = precice_read_data[:, 0]
                self._read_data[:, 1] = precice_read_data[:, 1]
                # z is the dead direction so it is supposed that the data is close to zero
                np.testing.assert_allclose(precice_read_data[:, 2], np.zeros_like(precice_read_data[:, 2]), )
                assert (np.sum(np.abs(precice_read_data[:, 2])) < 1e-10)
            else:
                raise Exception("Dimensions don't match.")
        else:
            raise Exception("Rank of function space is neither 0 nor 1")

    def write(self, write_function):
        """ Writes data to preCICE. Depending on the dimensions of the simulation (2D-3D Coupling, 2D-2D coupling or
        Scalar/Vector write function) write_data is first converted.

        :param write_function: FEniCS function
        """

        self._write_function_type = determine_function_type(write_function)
        assert (self._write_function_type in list(FunctionType))

        self._write_data = convert_fenics_to_precice(write_function, self._coupling_mesh_vertices)

        if self._write_function_type is FunctionType.SCALAR:
            self._interface.write_block_scalar_data(self._write_data_id, self._vertex_ids, self._write_data)
        elif self._write_function_type is FunctionType.VECTOR:
            if can_apply_2d_3d_coupling(self._fenics_dimensions, self._dimensions):
                # in 2d-3d coupling z dimension is set to zero
                precice_write_data = np.column_stack(self._write_data[:, 0], self._write_data[:, 1], np.zeros(self._n_vertices))

                assert (precice_write_data.shape[0] == self._n_vertices and
                        precice_write_data.shape[1] == self._dimensions)

                self._interface.write_block_vector_data(self._write_data_id, self._vertex_ids, precice_write_data)

            elif self._fenics_dimensions == self._dimensions:
                self._interface.write_block_vector_data(self._write_data_id, self._vertex_ids, self._write_data)
            else:
                raise Exception("Dimensions don't match.")
        else:
            raise Exception("Rank of function space is neither 0 nor 1")

    def initialize(self, coupling_subdomain, mesh, read_function, write_function, dimension=2):
        """Initializes remaining attributes. Called once, from the solver.

        :param write_function: FEniCS function for data to be written by this instance of coupling
        :param read_function: FEniCS function for data to be read by this instance of coupling
        :param coupling_subdomain: domain where coupling takes place
        :param mesh: fenics mesh
        :param dimension: problem dimension
        """

        self._fenics_dimensions = dimension

        if self._fenics_dimensions != self._dimensions:
            logger.warning(
                "fenics_dimension = {} and precice_dimension = {} do not match!".format(self._fenics_dimensions,
                                                                                        self._dimensions))
            if can_apply_2d_3d_coupling(self._fenics_dimensions, self._dimensions):
                logger.warning("2D-3D coupling will be applied. Z coordinates of all nodes will be set to zero.")
            else:
                raise Exception("fenics_dimension = {}, precice_dimension = {}. "
                                "No proper treatment for dimensional mismatch is implemented. Aborting!".format(
                    self._fenics_dimensions,
                    self._dimensions))

        self.set_coupling_mesh(mesh, coupling_subdomain)
        self._precice_tau = self._interface.initialize()

        # Set read_function and read_data
        self._read_function_type = determine_function_type(read_function)
        self._read_data = convert_fenics_to_precice(read_function, self._coupling_mesh_vertices)

        if self._interface.is_action_required(action_write_initial_data()):
            self.write(write_function)
            self._interface.mark_action_fulfilled(action_write_initial_data())

        self._interface.initialize_data()

        if self._interface.is_read_data_available():
            self.read()

        return self._precice_tau

    def set_coupling_mesh(self, mesh, subdomain):
        """Sets the coupling mesh. Called by initalize() function at the
        beginning of the simulation.
        :param subdomain: fenics subdomain on which coupling will be implemented
        :param mesh: fenics mesh
        :param: subdomain: subdomain which will be computed by this coupling instance
        """
        self._coupling_subdomain = subdomain
        self._mesh_fenics = mesh
        self._fenics_vertices, self._coupling_mesh_vertices, self._n_vertices \
            = extract_coupling_boundary_vertices(self._mesh_fenics, self._coupling_subdomain, self._fenics_dimensions, self._dimensions)
        self._vertex_ids = self._interface.set_mesh_vertices(self._mesh_id, self._coupling_mesh_vertices)

        """ Define a mapping between coupling vertices and their IDs in preCICE """
        id_mapping = dict()
        for i in range(self._n_vertices):
            id_mapping[self._fenics_vertices[i].global_index()] = self._vertex_ids[i]

        edge_vertex_ids1, edge_vertex_ids2 = extract_coupling_boundary_edges(self._mesh_fenics, self._coupling_subdomain, id_mapping)

        for i in range(len(edge_vertex_ids1)):
            assert (edge_vertex_ids1[i] != edge_vertex_ids2[i])
            self._interface.set_mesh_edge(self._mesh_id, edge_vertex_ids1[i], edge_vertex_ids2[i])

    def create_coupling_boundary_condition(self, function_space):
        """Creates the coupling boundary conditions using an actual implementation of CustomExpression."""
        x_vert, y_vert = extract_coupling_boundary_coordinates(self._coupling_mesh_vertices, self._fenics_dimensions, self._dimensions)

        try:  # works with dolfin 1.6.0
            coupling_bc_expression = self._my_expression(element=function_space.ufl_element())  # element information must be provided, else DOLFIN assumes scalar function
        except (TypeError, KeyError):  # works with dolfin 2017.2.0
            coupling_bc_expression = self._my_expression(element=function_space.ufl_element(), degree=0)

        coupling_bc_expression.set_boundary_data(self._read_data, x_vert, y_vert)

        return coupling_bc_expression

    def create_force_boundary_condition(self, Dirichlet_Boundary, function_space):
        """
        Initializes force-coupling via PointSource.
        This function only works for 2D-pseudo3D coupling.
        :param Dirichlet_Boundary:
        :param function_space: The Function Space used for the Test and Trial functions
        """
        self._has_force_boundary = True
        return get_forces_as_point_sources(Dirichlet_Boundary, function_space, self._coupling_mesh_vertices, self._read_data)

    def update_boundary_condition(self, coupling_bc_expression):
        x_vert, y_vert = extract_coupling_boundary_coordinates(self._coupling_mesh_vertices, self._fenics_dimensions, self._dimensions)
        if self._has_force_boundary:
            x_forces, y_forces = get_forces_as_point_sources()
        else:
            coupling_bc_expression.update_boundary_data(self._read_data, x_vert, y_vert)

    def store_checkpoint(self, u, t, n):
        """Stores the current solver state to a checkpoint.
        """
        logger.debug("Store checkpoint")
        self._checkpoint = SolverState(u, t, n)
        self._interface.mark_action_fulfilled(self.action_write_checkpoint())

    def retrieve_checkpoint(self):
        """Resets the solver's state to the checkpoint's state.
        """
        assert (not self._interface.is_time_window_complete())  # avoids invalid control flow
        logger.debug("Restore solver state")
        self._interface.mark_action_fulfilled(self.action_read_checkpoint())
        return self._checkpoint.get_state()

    def advance(self, dt):
        max_dt = self._interface.advance(dt)
        return max_dt

    def finalize(self):
        """Finalizes the coupling interface."""
        self._interface.finalize()

    def get_solver_name(self):
        """Returns name of this solver as defined in config file.
        :return: Solver name.
        """
        return self._solver_name

    def is_coupling_ongoing(self):
        """Determines whether simulation should continue. Called from the
        simulation loop in the solver.
        :return: True if the coupling is ongoing, False otherwise
        """
        return self._interface.is_coupling_ongoing()

    def is_timestep_complete(self):
        """
        :return: preCICE call if timestep is complete or not
        """
        return self._interface.is_time_window_complete()

    def is_action_required(self, action):
        """
        :param action:
        :return: preCICE call which returns true if provided action is required
        """
        return self._interface.is_action_required(action)

    def action_write_checkpoint(self):
        """
        :return: preCICE call to write checkpoint
        """
        return action_write_iteration_checkpoint()

    def action_read_checkpoint(self):
        """
        :return:
        """
        return action_read_iteration_checkpoint()
