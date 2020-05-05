"""
Adapter to FEniCS solver handles CustomExpression and initialization of the FEniCS adapter.
:raise ImportError: if PRECICE_ROOT is not defined
"""

import numpy as np
from .config import Config
import logging
import precice
from precice import action_write_initial_data, action_write_iteration_checkpoint, action_read_iteration_checkpoint
from .adapter_core import FunctionType, GeneralInterpolationExpression, ExactInterpolationExpression, \
    determine_function_type, \
    InterpolationType, can_apply_2d_3d_coupling, convert_fenics_to_precice, extract_coupling_boundary_vertices, \
    extract_coupling_boundary_edges, extract_coupling_boundary_coordinates, get_forces_as_point_sources
from .solverstate import SolverState
from fenics import MPI

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class Adapter:
    """
    Initializes the Adapter. Initalizer creates object of class Config (from
    config.py module) and creates an object of the preCICE Interface from the python
    bindings of the preCICE API.

    :ivar _config: object of class Config, which stores data from the JSON config file
    :ivar _interface: object of class Interface, which is used to call API functions

    :param adapter_config_filename: Name of .json config file
    """

    def __init__(self, adapter_config_filename='precice-adapter-config.json'):

        self._config = Config(adapter_config_filename)

        self._solver_name = self._config.get_solver_name()

        self._interface = precice.Interface(self._solver_name, self._config.get_config_file_name(),
                                            MPI.rank(MPI.comm_world), MPI.size(MPI.comm_world))
        self._dimensions = self._interface.get_dimensions()

        # FEniCS related quantities
        self._fenics_dimensions = None  # initialized later
        self._function_space = None  # initialized later

        # coupling mesh related quantities
        self._coupling_mesh_vertices = None  # initialized later
        self._mesh_name = self._config.get_coupling_mesh_name()
        self._mesh_id = self._interface.get_mesh_id(self._mesh_name)
        self._vertex_ids = None  # initialized later
        self._n_vertices = None  # initialized later

        # write data related quantities (write data is written by user from FEniCS to preCICE)
        self._write_data_name = self._config.get_write_data_name()
        self._write_data_id = self._interface.get_data_id(self._write_data_name, self._mesh_id)

        # read data related quantities (read data is read by use to FEniCS from preCICE)
        self._read_data_name = self._config.get_read_data_name()
        self._read_data_id = self._interface.get_data_id(self._read_data_name, self._mesh_id)
        self._read_function_type = None  # stores whether read function is scalar or vector valued
        self._read_function = None # Store the FEniCS function (initialized later)

        # Interpolation strategy as provided by the user
        self._my_expression = None  # initalized later

        # Solver state used by the Adapter internally to handle checkpointing
        self._checkpoint = None

        self._dss = None  # measure for boundary integral

        # Nodes with Dirichlet and Force-boundary
        self._Dirichlet_Boundary = None  # stores a dirichlet boundary (if provided)
        self._has_force_boundary = None  # stores whether force_boundary exists

        # Necessary flag in checkpoint storing function
        self._first_advance_done = None

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

    def create_coupling_expression(self):
        """
        Creates an object of class GeneralInterpolationExpression or ExactInterpolationExpression which does
        not carry any data. The adapter will hold this object till the coupling is on going.
        :return: coupling_expression: Return the reference to created FEniCS Expression object
        """
        try:  # works with dolfin 1.6.0
            coupling_expression = self._my_expression(element=self._function_space.ufl_element())  # element information must be provided, else DOLFIN assumes scalar function
        except (TypeError, KeyError):  # works with dolfin 2017.2.0
            coupling_expression = self._my_expression(element=self._function_space.ufl_element(), degree=0)

        return coupling_expression

    def update_coupling_expression(self, coupling_expression, data):
        """
        Updates a given coupling expression using a given data. The boundary data is updated.
        User needs to explicitly call this function in each time step
        :param coupling_expression: FEniCS Expression object
        :param data: Data used to update the boundary values in the coupling expression
        """
        x_vert, y_vert = extract_coupling_boundary_coordinates(self._coupling_mesh_vertices, self._fenics_dimensions,
                                                               self._dimensions)
        if self._n_vertices > 0:
            coupling_expression._is_empty = False
            coupling_expression.update_boundary_data(data, x_vert, y_vert)

        else:
            # having participants without coupling mesh nodes is only accepted for parallel runs
            assert (MPI.size(MPI.comm_world) > 1)
            # todo: this whole branch is currently a very ugly hack to make sure the coupling expression knows about
            # its dimension, even if no data is provided.
            coupling_expression._is_empty = True
            coupling_expression._vals = data
            coupling_expression._dimension = self._interface.get_dimensions()

    def create_point_sources(self, fixed_boundary, data):
        """
        Create point sources with reference to fixed boundary in a FSI simulation
        :return: lists containing point source values in X and Y directions respectively
        """
        self._Dirichlet_Boundary = fixed_boundary
        return get_forces_as_point_sources(self._Dirichlet_Boundary, self._function_space, self._coupling_mesh_vertices,
                                           data)

    def update_point_sources(self, data):
        """
        Update values of point sources using new data
        This function only works for 2D-pseudo3D coupling.
        :param data: 2D data from preCICE
        :return:
        """
        return get_forces_as_point_sources(self._Dirichlet_Boundary, self._function_space, self._coupling_mesh_vertices,
                                           data)

    def read(self):
        """
        Reads data from preCICE. Depending on the dimensions of the simulation (2D-3D Coupling, 2D-2D coupling or
        Scalar/Vector write function) read_data is converted.
        Note: For quasi 2D fenics in a 3D coupled simulation the y component of the vectors is deleted.
        :return: data read from preCICE in the form of a numpy array with the values like it is used by preCICE
        """
        assert (self._read_function_type in list(FunctionType))

        read_data = convert_fenics_to_precice(self._read_function, self._coupling_mesh_vertices)

        if self._n_vertices == 0:
            assert (MPI.size(MPI.comm_world) > 1)  # having participants without coupling mesh nodes is only accepted for parallel runs

        if self._n_vertices > 0:
            if self._read_function_type is FunctionType.SCALAR:
                read_data = self._interface.read_block_scalar_data(self._read_data_id, self._vertex_ids)
            elif self._read_function_type is FunctionType.VECTOR:
                if self._fenics_dimensions == self._dimensions:
                    read_data = self._interface.read_block_vector_data(self._read_data_id, self._vertex_ids)
                elif can_apply_2d_3d_coupling(self._fenics_dimensions, self._dimensions):
                    precice_read_data = self._interface.read_block_vector_data(self._read_data_id, self._vertex_ids)
                    read_data[:, 0] = precice_read_data[:, 0]
                    read_data[:, 1] = precice_read_data[:, 1]
                    # z is the dead direction so it is supposed that the data is close to zero
                    # np.testing.assert_allclose(precice_read_data[:, 2], np.zeros_like(precice_read_data[:, 2]), )
                    assert (np.sum(np.abs(precice_read_data[:, 2])) < 1e-10)
                else:
                    raise Exception("Dimensions don't match.")
            else:
                raise Exception("Rank of function space is neither 0 nor 1")
        else:  # even, if there are no vertices, we have to make sure that the coupling expression knows its dimension
            if self._read_function_type is FunctionType.SCALAR:
                read_data = np.empty(shape=(1))
            elif self._read_function_type is FunctionType.VECTOR:
                read_data = np.empty(shape=(self._interface.get_dimensions()))

        return read_data

    def write(self, write_function):
        """
        Writes data to preCICE. Depending on the dimensions of the simulation (2D-3D Coupling, 2D-2D coupling or
        Scalar/Vector write function) write_data is first converted.
        :param write_function: FEniCS function
        """
        write_function_type = determine_function_type(write_function)
        assert (write_function_type in list(FunctionType))

        print('{rank}: has n_vertices={n}'.format(rank=MPI.rank(MPI.comm_world), n=self._n_vertices))
        if self._n_vertices > 0:
            print('{rank}: alive'.format(rank=MPI.rank(MPI.comm_world)))
            write_function_type = determine_function_type(write_function)
            assert (write_function_type in list(FunctionType))
            print('{rank}: alive'.format(rank=MPI.rank(MPI.comm_world)))
            write_data = convert_fenics_to_precice(write_function, self._coupling_mesh_vertices)
            print('{rank}: alive'.format(rank=MPI.rank(MPI.comm_world)))
            if write_function_type is FunctionType.SCALAR:
                self._interface.write_block_scalar_data(self._write_data_id, self._vertex_ids, write_data)
            elif write_function_type is FunctionType.VECTOR:
                if can_apply_2d_3d_coupling(self._fenics_dimensions, self._dimensions):
                    # in 2d-3d coupling z dimension is set to zero
                    precice_write_data = np.column_stack(write_data[:, 0], write_data[:, 1], np.zeros(self._n_vertices))

                    assert (precice_write_data.shape[0] == self._n_vertices and
                            precice_write_data.shape[1] == self._dimensions)

                    self._interface.write_block_vector_data(self._write_data_id, self._vertex_ids, precice_write_data)
                elif self._fenics_dimensions == self._dimensions:
                    self._interface.write_block_vector_data(self._write_data_id, self._vertex_ids, write_data)
                else:
                    raise Exception("Dimensions don't match.")
            else:
                raise Exception("Rank of function space is neither 0 nor 1")

    def initialize(self, coupling_subdomain, mesh, dimension=2):
        """
        Initializes remaining attributes. Called once, from the solver.
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

        self._set_coupling_mesh(mesh, coupling_subdomain)
        precice_tau = self._interface.initialize()

        # Adapter is initialized but first advance is yet to be called
        self._first_advance_done = False

        return precice_tau

    def initialize_data(self, read_function, write_function, function_space):
        """
        Set initial conditions and boundary conditions
        :param read_function: FEniCS function
        :param write_function: FEniCS function
        :param function_space: FEniCS space
        :return:
        """
        self._read_function_type = determine_function_type(read_function)
        self._read_function = read_function
        self._function_space = function_space

        if self._interface.is_action_required(action_write_initial_data()):
            print('{rank} of {size}: is_action_required(action_write_initial_data())'.format(
                rank=MPI.rank(MPI.comm_world), size=MPI.size(MPI.comm_world)))
            self.write(write_function)
            self._interface.mark_action_fulfilled(action_write_initial_data())

        print('{rank} of {size}: initialize_data()'.format(
            rank=MPI.rank(MPI.comm_world), size=MPI.size(MPI.comm_world)))
        self._interface.initialize_data()

        print('{rank} of {size}: create_coupling_expression()'.format(
            rank=MPI.rank(MPI.comm_world), size=MPI.size(MPI.comm_world)))
        coupling_expression = self.create_coupling_expression()
        read_data = None

        if self._interface.is_read_data_available():
            print('{rank} of {size}: is_read_data_available()'.format(
                rank=MPI.rank(MPI.comm_world), size=MPI.size(MPI.comm_world)))
            read_data = self.read()

        return read_data

    def _set_coupling_mesh(self, mesh, subdomain):
        """
        Sets up the coupling mesh. This function is called by initalize() function at the
        beginning of the simulation.
        :param mesh: FEniCS mesh
        :param: subdomain: Part of FEniCS mesh which which will be computed by this participant
        """
        fenics_vertices, self._coupling_mesh_vertices, self._n_vertices \
            = extract_coupling_boundary_vertices(mesh, subdomain, self._fenics_dimensions, self._dimensions)

        if self._n_vertices > 0:
            self._vertex_ids = self._interface.set_mesh_vertices(self._mesh_id, self._coupling_mesh_vertices)
            """ Define a mapping between coupling vertices and their IDs in preCICE """
            id_mapping = dict()
            for i in range(self._n_vertices):
                id_mapping[fenics_vertices[i].global_index()] = self._vertex_ids[i]

            if MPI.size(MPI.comm_world) == 1:  # nearest-projection is only supported for non-parallel runs
                edge_vertex_ids1, edge_vertex_ids2 = extract_coupling_boundary_edges(mesh, subdomain, id_mapping)

                """ Set mesh edges in preCICE to allow nearest-projection mapping"""
                for i in range(len(edge_vertex_ids1)):
                    assert (edge_vertex_ids1[i] != edge_vertex_ids2[i])
                    self._interface.set_mesh_edge(self._mesh_id, edge_vertex_ids1[i], edge_vertex_ids2[i])
            else:
                print("Parallel FEniCS runs only support nearest neighbor.")

        if self._n_vertices == 0:
            assert (MPI.size(MPI.comm_world) > 1)  # having participants without coupling mesh nodes is only accepted for parallel runs   

    def store_checkpoint(self, user_u, t, n):
        """
        Defines an object of class SolverState which stores the current state of the variable and the time stamp
        :param user_u: Variable being computed
        :param t: Physical time
        :param n: Simulation iteration counter
        """
        if self._first_advance_done:
            assert (self.is_time_window_complete())

        logger.debug("Store checkpoint")
        my_u = user_u.copy()
        assert (my_u != user_u)  # wrt to pointer
        self._checkpoint = SolverState(my_u, t, n)
        self._interface.mark_action_fulfilled(self.action_write_checkpoint())

    def retrieve_checkpoint(self):
        """
        Resets the solver's state to the checkpoint's state.
        :return: State stored as a checkpoint
        """
        assert (not self.is_time_window_complete())  # avoids invalid control flow
        logger.debug("Restore solver state")
        self._interface.mark_action_fulfilled(self.action_read_checkpoint())
        return self._checkpoint.get_state()

    def advance(self, dt):
        """
        Advances coupling
        :param dt: Current used time step
        :return: time step recommended by preCICE
        """
        self._first_advance_done = True
        max_dt = self._interface.advance(dt)
        return max_dt

    def finalize(self):
        """
        Finalizes the coupling interface. To be called at the end of the simulation
        """
        self._interface.finalize()

    def get_solver_name(self):
        """
        :return: Solver name
        """
        return self._solver_name

    def is_coupling_ongoing(self):
        """
        :return: True if the coupling is ongoing, False otherwise
        """
        return self._interface.is_coupling_ongoing()

    def is_time_window_complete(self):
        """
        :return: True if implicit coupling has converged, False otherwise
        """
        return self._interface.is_time_window_complete()

    def is_action_required(self, action):
        """
        :param action: preCICE action
        :return: preCICE call which returns true if provided action is required
        """
        return self._interface.is_action_required(action)

    def action_write_checkpoint(self):
        """
        :return: True if checkpoint needs to be written, False otherwise
        """
        return action_write_iteration_checkpoint()

    def action_read_checkpoint(self):
        """
        :return: True if checkpoint needs to be read, False otherwise
        """
        return action_read_iteration_checkpoint()
