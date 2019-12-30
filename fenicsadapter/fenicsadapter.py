""" This module handles CustomExpression and initialization of the FEniCS adapter.
:raise ImportError: if PRECICE_ROOT is not defined
"""

import dolfin
from dolfin import FacetNormal, dot
import numpy as np
from .config import Config
from .checkpointing import Checkpoint
import logging
import precice
from precice import action_write_initial_data, action_write_iteration_checkpoint, action_read_iteration_checkpoint
from .adapter_core import AdapterCore, FunctionType, GeneralInterpolationExpression, determine_function_type
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

        # Temporarily hard-coding interpolation strategy. Need to provide user with the appropriate choice
        self._my_expression = GeneralInterpolationExpression

        # checkpointing
        self._checkpoint = Checkpoint()

        # function space
        self._function_space = None
        self._dss = None  # measure for boundary integral

        # Nodes with Dirichlet and Force-boundary
        self._Dirichlet_Boundary = None  # stores a dirichlet boundary (if provided)
        self._has_force_boundary = None  # stores whether force_boundary exists

        # Adapter core
        self._CoreObject = None  # Adapter core object. Initialized later

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

            elif self._CoreObject.can_apply_2d_3d_coupling():
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

        self._write_data = self._CoreObject.convert_fenics_to_precice(write_function)

        if self._write_function_type is FunctionType.SCALAR:
            self._interface.write_block_scalar_data(self._write_data_id, self._vertex_ids, self._write_data)
        elif self._write_function_type is FunctionType.VECTOR:
            if self._CoreObject.can_apply_2d_3d_coupling():
                # in 2d-3d coupling z dimension is set to zero
                precice_write_data = np.column_stack((self._write_data[:, 0], self._write_data[:, 1], np.zeros(self._n_vertices)))

                assert (precice_write_data.shape[0] == self._n_vertices and
                        precice_write_data.shape[1] == self._dimensions)

                self._interface.write_block_vector_data(self._write_data_id, self._vertex_ids, precice_write_data)

            elif self._fenics_dimensions == self._dimensions:
                self._interface.write_block_vector_data(self._write_data_id, self._vertex_ids, self._write_data)
            else:
                raise Exception("Dimensions don't match.")
        else:
            raise Exception("Rank of function space is neither 0 nor 1")

    def initialize(self, coupling_subdomain, mesh, u_n, read_function, write_function, dimension=2, t=0, n=0):
        """Initializes remaining attributes. Called once, from the solver.

        :param write_function:
        :param read_function:
        :param coupling_subdomain: domain where coupling takes place
        :param mesh: fenics mesh
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
            if self._CoreObject.can_apply_2d_3d_coupling():
                logger.warning("2D-3D coupling will be applied. Z coordinates of all nodes will be set to zero.")
            else:
                raise Exception("fenics_dimension = {}, precice_dimension = {}. "
                                "No proper treatment for dimensional mismatch is implemented. Aborting!".format(
                    self._fenics_dimensions,
                    self._dimensions))

        # Initialize an object of the Adapter core to access all functionality
        self._CoreObject = AdapterCore(self._dimensions, self._fenics_dimensions, mesh, coupling_subdomain)

        # Set read_function and read_data
        self._read_function_type = determine_function_type(read_function)
        self._read_data = self._CoreObject.convert_fenics_to_precice(read_function)

        self.set_coupling_mesh(mesh, coupling_subdomain)
        self._precice_tau = self._interface.initialize()

        if self._interface.is_action_required(precice.action_write_initial_data()):
            self.write(write_function)
            self._interface.fulfilled_action(precice.action_write_initial_data())

        self._interface.initialize_data()

        if self._interface.is_read_data_available():
            self._read_data = self.read()

        if self._interface.is_action_required(precice.action_write_iteration_checkpoint()):
            initial_state = SolverState(u_n, t, n)
            self.save_solver_state_to_checkpoint(initial_state)

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
        self._fenics_vertices, self._coupling_mesh_vertices, self._n_vertices = self._CoreObject.extract_coupling_boundary_vertices()
        self._vertex_ids = self._interface.set_mesh_vertices(self._mesh_id, self._coupling_mesh_vertices)

        """ Define a mapping between coupling vertices and their IDs in preCICE """
        id_mapping = dict()
        for i in range(self._n_vertices):
            id_mapping[self._fenics_vertices[i].global_index()] = self._vertex_ids[i]

        self._edge_vertex_ids1, self._edge_vertex_ids2 = self._CoreObject.extract_coupling_boundary_edges(id_mapping)

        for i in range(len(self._edge_vertex_ids1)):
            assert (self._edge_vertex_ids1[i] != self._edge_vertex_ids2[i])
            self._interface.set_mesh_edge(self._mesh_id, self._edge_vertex_ids1[i], self._edge_vertex_ids2[i])

    def create_coupling_dirichlet_boundary_condition(self, function_space):
        """Creates the coupling Dirichlet boundary conditions using
        create_coupling_boundary_condition() method.

        :return: dolfin.DirichletBC()
        """
        self._Dirichlet_Boundary = True

        self._coupling_bc_expression = self._CoreObject.create_coupling_boundary_condition(self._my_expression, self._read_data
                                                                                           , function_space)
        return dolfin.DirichletBC(function_space, self._coupling_bc_expression, self._coupling_subdomain)

    def create_coupling_neumann_boundary_condition(self, test_functions, function_space=None, boundary_marker=None):
        """Creates the coupling Neumann boundary conditions using
        create_coupling_boundary_condition() method.

        :return: expression in form of integral: g*v*ds. (see e.g. p. 83ff
        Langtangen, Hans Petter, and Anders Logg. "Solving PDEs in Python The
        FEniCS Tutorial Volume I." (2016).)
        """
        if not function_space:
            function_space = test_functions.function_space()
        else:
            function_space = function_space
        self._coupling_bc_expression = self._CoreObject.create_coupling_boundary_condition(self._my_expression, self._read_data
                                                                                           , function_space)
        if not boundary_marker:  # there is only 1 Neumann-BC which is at the coupling boundary -> integration over whole boundary
            if self._coupling_bc_expression.is_scalar_valued():
                return test_functions * self._coupling_bc_expression * dolfin.ds  # this term has to be added to weak form to add a Neumann BC (see e.g. p. 83ff Langtangen, Hans Petter, and Anders Logg. "Solving PDEs in Python The FEniCS Tutorial Volume I." (2016).)
            elif self._coupling_bc_expression.is_vector_valued():
                n = FacetNormal(self._mesh_fenics)
                return -test_functions * dot(n, self._coupling_bc_expression) * dolfin.ds
            else:
                raise Exception("invalid!")
        else:  # For multiple Neumann BCs integration should only be performed over the respective domain.
            # TODO: fix the problem here
            raise Exception("Boundary markers are not implemented yet")
            return dot(self._coupling_bc_expression, test_functions) * self.dss(boundary_marker)

    def create_force_boundary_condition(self, Dirichlet_Boundary, function_space):
        """
        Initializes force-coupling via PointSource.

        This function only works for 2D-pseudo3D coupling.

        :param Dirichlet_Boundary:
        :param function_space: The Function Space used for the Test and Trial functions
        """
        self._has_force_boundary = True

        return AdapterCore.get_forces_as_point_sources(Dirichlet_Boundary, function_space)

    def update_boundary_condition(self):
        x_vert, y_vert = self._CoreObject.extract_coupling_boundary_coordinates()
        if self._has_force_boundary:
            x_forces, y_forces = self._CoreObject.get_forces_as_point_sources()
        else:
            self._coupling_bc_expression.update_boundary_data(self._read_data, x_vert, y_vert)

    def is_coupling_ongoing(self):
        """Determines whether simulation should continue. Called from the
        simulation loop in the solver.
        :return: True if the coupling is ongoing, False otherwise
        """
        return self._interface.is_coupling_ongoing()

    def initialize_solver_state(self, u_n, t, n):
        """Initalizes the solver state before coupling starts in each iteration
        :param u_n:
        :param t:
        :param n:
        """
        state = SolverState(u_n, t, n)
        logger.debug("Solver state is initialized")
        return state

    def restore_solver_state_from_checkpoint(self, state):
        """Resets the solver's state to the checkpoint's state.
        :param state: current state of the FEniCS solver
        """
        logger.debug("Restore solver state")
        state.update(self._checkpoint.get_state())
        self._interface.fulfilled_action(precice.action_read_iteration_checkpoint())

    def advance_solver_state(self, state, u_np1, dt):
        """Advances the solver's state by one timestep. Also advances coupling state
        :param state: old state
        :param u_np1: new value
        :param dt: timestep size
        :return: maximum time step value recommended by preCICE
        """
        logger.debug("Advance solver state")
        logger.debug("old state: t={time}".format(time=state.t))
        state.update(SolverState(u_np1, state.t + dt, state.n + 1))
        logger.debug("new state: t={time}".format(time=state.t))

    def advance_coupling(self, dt, fenics_dt):
        max_dt = self._interface.advance(dt)
        max_dt = np.min(max_dt, fenics_dt)
        return max_dt

    def save_solver_state_to_checkpoint(self, state):
        """Writes given solver state to checkpoint.
        :param state: state being saved as checkpoint
        """
        logger.debug("Save solver state")
        self._checkpoint.write(state)
        self._interface.fulfilled_action(precice.action_write_iteration_checkpoint())

    def finalize(self):
        """Finalizes the coupling interface."""
        self._interface.finalize()

    def get_solver_name(self):
        """Returns name of this solver as defined in config file.
        :return: Solver name.
        """
        return self._solver_name

    def is_timestep_complete(self):
        """
        :return: preCICE call if timestep is complete or not
        """
        return self._interface.is_timestep_complete()

    def is_action_required(self, action):
        """
        :return: preCICE call which returns true if provided action is required
        """
        return self._interface.is_action_required(action)

    def write_checkpoint(self):
        """
        :return: preCICE call to write checkpoint
        """
        return action_write_iteration_checkpoint()

    def restore_to_checkpoint(self):
        """
        :return:
        """
        return action_read_iteration_checkpoint()
