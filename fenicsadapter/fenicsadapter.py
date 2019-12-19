""" This module handles CustomExpression and initialization of the FEniCS adapter.
:raise ImportError: if PRECICE_ROOT is not defined
"""

import numpy as np
from .config import Config
from .checkpointing import Checkpoint
import logging
import precice
from .adapter_core import FunctionType, extract_coupling_boundary_vertices, extract_coupling_boundary_edges, GeneralInterpolationExpression\
    , set_read_field, set_write_field
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

        self._interface = precice.Interface(self._solver_name, 0, 1)
        self._interface.configure(self._config.get_config_file_name())
        self._dimensions = self._interface.get_dimensions()

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

        # write data related quantities (write data is written by this solver to preCICE)
        self._write_data_name = self._config.get_write_data_name()
        self._write_data_id = self._interface.get_data_id(self._write_data_name, self._mesh_id)
        self._write_data = None  # a numpy 1D array with the values like it is used by precice (The 2D-format of values is (d0x, d0y, d1x, d1y, ..., dnx, dny) The 3D-format of values is (d0x, d0y, d0z, d1x, d1y, d1z, ..., dnx, dny, dnz))
        self._write_function_type = None  # stores whether write function is scalar or vector valued

        # read data related quantities (read data is read by this solver from preCICE)
        self._read_data_name = self._config.get_read_data_name()
        self._read_data_id = self._interface.get_data_id(self._read_data_name, self._mesh_id)
        self._read_data = None  # a numpy 1D array with the values like it is used by precice (The 2D-format of values is (d0x, d0y, d1x, d1y, ..., dnx, dny) The 3D-format of values is (d0x, d0y, d0z, d1x, d1y, d1z, ..., dnx, dny, dnz))
        self._read_function_type = None  # stores whether read function is scalar or vector valued

        # numerics
        self._precice_tau = None
        # Temporarily hard-coding interpolation strategy. Need to provide
        self._my_expression = GeneralInterpolationExpression

        # checkpointing
        self._checkpoint = Checkpoint()

        # function space
        self._function_space = None
        self._dss = None  # measure for boundary integral

        # Nodes with Dirichlet and Force-boundary
        self._Dirichlet_Boundary = None  # stores a dirichlet boundary (if provided)
        self._has_force_boundary = None  # stores whether force_boundary exists

        def read_data():
            """ Reads data from preCICE. Depending on the dimensions of the simulation (2D-3D Coupling, 2D-2D coupling or
            Scalar/Vector write function) read_data is converted.

            Note: For quasi 2D fenics in a 3D coupled simulation the y component of the vectors is deleted.
            """

            assert (self._read_function_type in list(FunctionType))

            if self._read_function_type is FunctionType.SCALAR:
                self._read_data = self._interface.read_block_scalar_data(self._read_data_id, self._vertex_ids)

            elif self._read_function_type is FunctionType.VECTOR:
                if self._fenics_dimensions == self._dimensions:
                    self._read_data = self._interface.read_block_vector_data(self._read_data_id, self._vertex_ids)

                elif self._can_apply_2d_3d_coupling():
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

        def write_data():
            """ Writes data to preCICE. Depending on the dimensions of the simulation (2D-3D Coupling, 2D-2D coupling or
            Scalar/Vector write function) write_data is first converted.
            """

            assert (self._write_function_type in list(FunctionType))

            if self._write_function_type is FunctionType.SCALAR:
                self._interface.write_block_scalar_data(self._write_data_id, self._vertex_ids, self._write_data)
            elif self._write_function_type is FunctionType.VECTOR:
                if self._can_apply_2d_3d_coupling():
                    # in 2d-3d coupling z dimension is set to zero
                    precice_write_data = np.column_stack((self._write_data[:, 0],
                                                          self._write_data[:, 1],
                                                          np.zeros(self._n_vertices)))

                    assert (precice_write_data.shape[0] == self._n_vertices and
                            precice_write_data.shape[1] == self._dimensions)

                    self._interface.write_block_vector_data(self._write_data_id, self._vertex_ids, precice_write_data)

                elif self._fenics_dimensions == self._dimensions:
                    self._interface.write_block_vector_data(self._write_data_id, self._vertex_ids, self._write_data)
                else:
                    raise Exception("Dimensions don't match.")
            else:
                raise Exception("Rank of function space is neither 0 nor 1")

        def initialize(self, coupling_subdomain, mesh, read_field, write_field,
                       u_n, dimension=2, t=0, n=0, dirichlet_boundary=None):
            """Initializes remaining attributes. Called once, from the solver.

            :param coupling_subdomain: domain where coupling takes place
            :param mesh: fenics mesh
            :param read_field: function applied on the read field
            :param write_field: function applied on the write field
            :param u_n: initial data for solution
            :param dimension: problem dimension
            :param t: starting time
            :param n: time step n
            :param coupling_marker: boundary marker, can be used for coupling, multiple Neumann boundary conditions are applied
            """
            self._fenics_dimensions = dimension

            if self._fenics_dimensions != self._dimensions:
                logger.warning(
                    "fenics_dimension = {} and precice_dimension = {} do not match!".format(self._fenics_dimensions,
                                                                                            self._dimensions))
                if self._can_apply_2d_3d_coupling():
                    logger.warning("2D-3D coupling will be applied. Z coordinates of all nodes will be set to zero.")
                else:
                    raise Exception("fenics_dimension = {}, precice_dimension = {}. "
                                    "No proper treatment for dimensional mismatch is implemented. Aborting!".format(
                        self._fenics_dimensions,
                        self._dimensions))
            if dirichlet_boundary is not None:
                self._Dirichlet_Boundary = dirichlet_boundary

            set_coupling_mesh(mesh, coupling_subdomain)
            set_read_field(read_field)
            set_write_field(write_field)
            self._precice_tau = self._interface.initialize()

            if self._interface.is_action_required(precice.action_write_initial_data()):
                write_data()
                self._interface.fulfilled_action(precice.action_write_initial_data())

            self._interface.initialize_data()

            if self._interface.is_read_data_available():
                read_data()

            if self._interface.is_action_required(precice.action_write_iteration_checkpoint()):
                initial_state = SolverState(u_n, t, n)
                self._save_solver_state_to_checkpoint(initial_state)

            return self._precice_tau

        def set_coupling_mesh(mesh, subdomain):
            """Sets the coupling mesh. Called by initalize() function at the
            beginning of the simulation.
            """
            self._coupling_subdomain = subdomain
            self._mesh_fenics = mesh
            self._fenics_vertices, self._coupling_mesh_vertices, self._n_vertices = extract_coupling_boundary_vertices()
            self._vertex_ids = self._interface.set_mesh_vertices(self._mesh_id, self._coupling_mesh_vertices)

            """ Define a mapping between coupling vertices and their IDs in precice"""
            id_mapping = dict()
            for i in range(self._n_vertices):
                id_mapping[self._fenics_vertices[i].global_index()] = self._vertex_ids[i]

            self._edge_vertex_ids1, self._edge_vertex_ids2 = extract_coupling_boundary_edges(id_mapping)

            for i in range(len(self._edge_vertex_ids1)):
                assert (self._edge_vertex_ids1[i] != self._edge_vertex_ids2[i])
                self._interface.set_mesh_edge(self._mesh_id, self._edge_vertex_ids1[i], self._edge_vertex_ids2[i])

        def is_coupling_ongoing():
            """Determines whether simulation should continue. Called from the
            simulation loop in the solver.
            :return: True if the coupling is ongoing, False otherwise
            """
            return self._interface.is_coupling_ongoing()

        def restore_solver_state_from_checkpoint(state):
            """Resets the solver's state to the checkpoint's state.
            :param state: current state of the FEniCS solver
            """
            logger.debug("Restore solver state")
            state.update(self._checkpoint.get_state())
            self._interface.fulfilled_action(precice.action_read_iteration_checkpoint())

        def advance_solver_state(state, u_np1, dt):
            """Advances the solver's state by one timestep.
            :param state: old state
            :param u_np1: new value
            :param dt: timestep size
            :return:
            """
            logger.debug("Advance solver state")
            logger.debug("old state: t={time}".format(time=state.t))
            state.update(SolverState(u_np1, state.t + dt, state.n + 1))
            logger.debug("new state: t={time}".format(time=state.t))

        def save_solver_state_to_checkpoint(state):
            """Writes given solver state to checkpoint.
            :param state: state being saved as checkpoint
            """
            logger.debug("Save solver state")
            self._checkpoint.write(state)
            self._interface.fulfilled_action(precice.action_write_iteration_checkpoint())

        def finalize():
            """Finalizes the coupling interface."""
            self._interface.finalize()

        def get_solver_name():
            """Returns name of this solver as defined in config file.
            :return: Solver name.
            """
            return self._solver_name

