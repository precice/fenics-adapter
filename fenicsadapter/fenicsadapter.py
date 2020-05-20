"""
Adapter module to provide a FEniCS Adapter to simulate partitioned coupled problems with FEniCS as a coupling participant
:raise ImportError: if PRECICE_ROOT is not defined
"""
import numpy as np
from .config import Config
import logging
import precice
from precice import action_write_initial_data, action_write_iteration_checkpoint, action_read_iteration_checkpoint
from .adapter_core import FunctionType, determine_function_type, convert_fenics_to_precice, \
    get_coupling_boundary_vertices, get_coupling_boundary_edges, get_forces_as_point_sources
from .expression_core import GeneralInterpolationExpression, ExactInterpolationExpression
from .solverstate import SolverState
from warnings import warn

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class Adapter:
    """
    Main class for the FEniCS Adapter.
    """
    def __init__(self, adapter_config_filename='precice-adapter-config.json'):
        """
        Constructor of Adapter class.

        Parameters
        ----------
        adapter_config_filename : string
            Name of the JSON adapter configuration file (to be provided by the user)
        """

        self._config = Config(adapter_config_filename)

        self._interface = precice.Interface(self._config.get_solver_name(), self._config.get_config_file_name(), 0, 1)

        # FEniCS related quantities
        self._fenics_dimensions = None
        self._function_space = None  # initialized later

        # coupling mesh related quantities
        self._coupling_mesh_vertices = None  # initialized later
        self._vertex_ids = None  # initialized later
        self._n_vertices = None  # initialized later

        # read data related quantities (read data is read by use to FEniCS from preCICE)
        self._read_function_type = None  # stores whether read function is scalar or vector valued

        # Interpolation strategy as provided by the user
        if self._config.get_interpolation_expression_type() == "cubic_spline":
            self._my_expression = ExactInterpolationExpression
            print("Using cubic spline interpolation")
        elif self._config.get_interpolation_expression_type() == "rbf":
            self._my_expression = GeneralInterpolationExpression
            print("Using RBF interpolation")
        else:
            warn("No valid interpolation strategy entered. It is assumed that the user does "
                 "not wish to use FEniCS Expressions on the coupling boundary.")

        # Solver state used by the Adapter internally to handle checkpointing
        self._checkpoint = None

        self._dss = None  # measure for boundary integral

        # Nodes with Dirichlet and Force-boundary
        self._Dirichlet_Boundary = None  # stores a dirichlet boundary (if provided)
        self._has_force_boundary = None  # stores whether force_boundary exists

        # Necessary flag in checkpoint storing function
        self._first_advance_done = False

        # Flag to see if 2D - 3D coupling needs to be applied
        self._apply_2d_3d_coupling = False

    def create_coupling_expression(self, data):
        """
        Creates an object of class GeneralInterpolationExpression or ExactInterpolationExpression which does
        not carry any data. The adapter will hold this object till the coupling is on going.

        Parameters
        ----------
        data : numpy.ndarray
            The coordinates of the vertices. Coordinates of vertices are stored in a
            numpy array [N x D] where N = number of vertices and D = dimensions of geometry

        Returns
        -------
        coupling_expression : object of FEniCS class CustomExpression
            Reference to object of FEniCS class CustomExpression.

        """
        try:  # works with dolfin 1.6.0
            coupling_expression = self._my_expression(
                element=self._function_space.ufl_element())  # element information must be provided, else DOLFIN assumes scalar function
        except (TypeError, KeyError):  # works with dolfin 2017.2.0
            coupling_expression = self._my_expression(element=self._function_space.ufl_element(), degree=0)

        coupling_expression.update_boundary_data(data, self._coupling_mesh_vertices[:, 0], self._coupling_mesh_vertices[:, 1])

        return coupling_expression

    def update_coupling_expression(self, coupling_expression, data):
        """
        Updates a given coupling expression using a given data. The boundary data is updated.
        User needs to explicitly call this function in each time step.

        Parameters
        ----------
        coupling_expression : object of class FEniCS CustomExpression
            Reference to object of FEniCS class CustomExpression.
        data : numpy.ndarray
            The coordinates of the vertices. Coordinates of vertices are stored in a
            numpy array [N x D] where N = number of vertices and D = dimensions of geometry.
        """
        coupling_expression.update_boundary_data(data, self._coupling_mesh_vertices[:, 0], self._coupling_mesh_vertices[:, 1])

    def create_point_sources(self, fixed_boundary, data):
        """
        Create point sources with reference to fixed boundary in a FSI simulation.

        Parameters
        ----------
        fixed_boundary : FEniCS domain
            FEniCS domain consisting of a fixed boundary condition. For example in FSI cases usually the solid body
            is fixed at one end.
        data : PointSource
            FEniCS PointSource data.

        Returns
        -------
        x_forces : list
            List containing X component of forces with reference to respective point sources on the coupling interface.
        y_forces : list
            List containing Y component of forces with reference to respective point sources on the coupling interface.
        """
        self._Dirichlet_Boundary = fixed_boundary
        return get_forces_as_point_sources(self._Dirichlet_Boundary, self._function_space, self._coupling_mesh_vertices,
                                           data)

    def update_point_sources(self, data):
        """
        Update values of point sources using data. This function only works for 2D-pseudo3D coupling.

        Parameters
        ----------
        data : PointSource
            FEniCS PointSource data.

        Returns
        -------
        x_forces : list
            List containing X component of forces with reference to respective point sources on the coupling interface.
        y_forces : list
            List containing Y component of forces with reference to respective point sources on the coupling interface.
        """
        return get_forces_as_point_sources(self._Dirichlet_Boundary, self._function_space, self._coupling_mesh_vertices,
                                           data)

    def read(self):
        """
        Read data from preCICE. Data is generated in an appropriate form depending on the dimensions of the
        simulation (2D-3D Coupling, 2D-2D coupling or Scalar/Vector write function).
        Note: For quasi 2D fenics in a 3D coupled simulation the y component of the vectors is deleted.

        Returns
        -------
        read_data : numpy.ndarray
            Contains the read data.
        """
        assert (self._read_function_type in list(FunctionType))

        read_data = None

        read_data_id = self._interface.get_data_id(self._config.get_read_data_name(),
                                                   self._interface.get_mesh_id(self._config.get_coupling_mesh_name()))

        if self._read_function_type is FunctionType.SCALAR:
            read_data = self._interface.read_block_scalar_data(read_data_id, self._vertex_ids)
        elif self._read_function_type is FunctionType.VECTOR:
            if self._fenics_dimensions == self._interface.get_dimensions():
                read_data = self._interface.read_block_vector_data(read_data_id, self._vertex_ids)
            elif self._apply_2d_3d_coupling:
                precice_read_data = self._interface.read_block_vector_data(read_data_id, self._vertex_ids)
                read_data[:, 0] = precice_read_data[:, 0]
                read_data[:, 1] = precice_read_data[:, 1]
                # z is the dead direction so it is supposed that the data is close to zero
                # np.testing.assert_allclose(precice_read_data[:, 2], np.zeros_like(precice_read_data[:, 2]))
                assert (np.sum(np.abs(precice_read_data[:, 2])) < 1e-10)
            else:
                raise Exception("Dimensions do not match.")
        else:
            raise Exception("Rank of function space is neither 0 nor 1")

        return read_data

    def write(self, write_function):
        """
        Writes data to preCICE. Depending on the dimensions of the simulation (2D-3D Coupling, 2D-2D coupling or
        Scalar/Vector write function) write_data is first converted into a format needed for preCICE.

        Parameters
        ----------
        write_function : FEniCS function
            A FEniCS function consisting of the data which this participant will write to preCICE in every time step.
        """
        write_function_type = determine_function_type(write_function)
        assert (write_function_type in list(FunctionType))

        write_data = convert_fenics_to_precice(write_function, self._coupling_mesh_vertices)

        write_data_id = self._interface.get_data_id(self._config.get_write_data_name(),
                                                    self._interface.get_mesh_id(self._config.get_coupling_mesh_name()))

        if write_function_type is FunctionType.SCALAR:
            self._interface.write_block_scalar_data(write_data_id, self._vertex_ids, write_data)
        elif write_function_type is FunctionType.VECTOR:
            if self._apply_2d_3d_coupling:
                # in 2d-3d coupling z dimension is set to zero
                precice_write_data = np.column_stack((write_data[:, 0], write_data[:, 1], np.zeros(self._n_vertices)))
                assert (precice_write_data.shape[0] == self._n_vertices and
                        precice_write_data.shape[1] == self._interface.get_dimensions())
                self._interface.write_block_vector_data(write_data_id, self._vertex_ids, precice_write_data)
            elif self._fenics_dimensions == self._interface.get_dimensions():
                self._interface.write_block_vector_data(write_data_id, self._vertex_ids, write_data)
            else:
                raise Exception("Dimensions don't match.")
        else:
            raise Exception("Rank of function space is neither 0 nor 1")

    def initialize(self, coupling_subdomain, mesh, read_function, function_space, dimensions=2):
        """
        Initializes the coupling interface and sets up the mesh in preCICE.

        Parameters
        ----------
        coupling_subdomain : FEniCS Domain
            Part of FEniCS Mesh which is the coupling interface.
        mesh : FEniCS Mesh
            FEniCS mesh of the complete region.
        read_function : FEniCS Function
            FEniCS function consisting of the data which this participant will read from preCICE in every time step.
        function_space : FEniCS Function Space
            Function space on which the finite element formulation of the problem lives.
        dimensions : int
            Dimensions of the problem as defined in FEniCS.

        Returns
        -------
        dt : double
            Recommended time step value from preCICE.
        """

        self._fenics_dimensions = dimensions

        if dimensions != self._interface.get_dimensions():
            logger.warning("fenics_dimension = {} and precice_dimension = {} do not match!".format(
                dimensions, self._interface.get_dimensions()))
            """ 
            In certain situations a 2D-3D coupling is applied. This means that the y-dimension of data and nodes
            received from preCICE is ignored. If FEniCS sends data to preCICE, the y-dimension of data and node 
            coordinates is set to zero.
            """
            if dimensions == 2 and self._interface.get_dimensions() == 3:
                logger.warning("2D-3D coupling will be applied. Z coordinates of all nodes will be set to zero.")
                self._apply_2d_3d_coupling = True
            else:
                raise Exception("fenics_dimension = {}, precice_dimension = {}. "
                                "No proper treatment for dimensional mismatch is implemented. Aborting!".format(
                    dimensions, self._interface.get_dimensions()))

        fenics_vertices, self._coupling_mesh_vertices, self._n_vertices \
            = get_coupling_boundary_vertices(mesh, coupling_subdomain, dimensions, self._interface.get_dimensions())

        """ Set up mesh in preCICE """
        self._vertex_ids = self._interface.set_mesh_vertices(self._interface.get_mesh_id(
            self._config.get_coupling_mesh_name()), self._coupling_mesh_vertices)

        """ Define a mapping between coupling vertices and their IDs in preCICE """
        id_mapping = dict()
        for i in range(self._n_vertices):
            id_mapping[fenics_vertices[i].global_index()] = self._vertex_ids[i]
        edge_vertex_ids1, edge_vertex_ids2 = get_coupling_boundary_edges(mesh, coupling_subdomain, id_mapping)

        """ Set mesh edges in preCICE to allow nearest-projection mapping"""
        for i in range(len(edge_vertex_ids1)):
            assert (edge_vertex_ids1[i] != edge_vertex_ids2[i])
            self._interface.set_mesh_edge(self._interface.get_mesh_id(self._config.get_coupling_mesh_name()),
                                          edge_vertex_ids1[i], edge_vertex_ids2[i])

        """ Set read functionality parameters """
        self._read_function_type = determine_function_type(read_function)
        self._function_space = function_space

        return self._interface.initialize()

    def initialize_data(self, write_function):
        """
        Set non-standard initial conditions and boundary conditions

        Parameters
        ----------
        write_function : FEniCS Function
            FEniCS function consisting of the data which this participant will write to preCICE in every time step.

        Returns
        -------
        read_data = numpy.ndarray
            Contains data read from preCICE.

        """

        if self._interface.is_action_required(action_write_initial_data()):
            self.write(write_function)
            self._interface.mark_action_fulfilled(action_write_initial_data())

        self._interface.initialize_data()

        read_data = None
        if self._interface.is_read_data_available():
            read_data = self.read()

        return read_data

    def store_checkpoint(self, user_u, t, n):
        """
        Defines an object of class SolverState which stores the current state of the variable and the time stamp.

        Parameters
        ----------
        user_u : FEniCS Function
            Current state of the physical variable of interest for this participant.
        t : double
            Current simulation time.
        n : int
            Current time window (iteration) number.
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
        Resets the FEniCS participant state to the state of the stored checkpoint.

        Returns
        -------
        u : FEniCS Function
            Current state of the physical variable of interest for this participant.
        t : double
            Current simulation time.
        n : int
            Current time window (iteration) number.
        """
        assert (not self.is_time_window_complete())  # avoids invalid control flow
        logger.debug("Restore solver state")
        self._interface.mark_action_fulfilled(self.action_read_checkpoint())
        return self._checkpoint.get_state()

    def advance(self, dt):
        """
        Advances coupling in preCICE.

        Parameters
        ----------
        dt : double
            Time step value used for the last iteration.

        Returns
        -------
        max_dt : double
            Recommended time step value from preCICE.
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
        Returns
        -------
        solver_name : string
            Name of the solver.
        """
        return self._config.get_solver_name()

    def is_coupling_ongoing(self):
        """
        Tag to check if coupling is still going on.

        Returns
        -------
        tag : bool
            True if coupling is still going on and False if coupling has finished.
        """
        return self._interface.is_coupling_ongoing()

    def is_time_window_complete(self):
        """
        Tag to check if implicit iteration has converged.

        Returns
        -------
        tag : bool
            True if implicit coupling in the time window has converged and False if not converged yet.
        """
        return self._interface.is_time_window_complete()

    def is_action_required(self, action):
        """
        Tag to check if a particular preCICE action is required.

        Parameters
        ----------
        action : string
            Name of the preCICE action.

        Returns
        -------
        tag : bool
            True if action is required and False if action is not required.
        """
        return self._interface.is_action_required(action)

    def action_write_checkpoint(self):
        """
        Get name of action to convey to preCICE that a checkpoint has been written.

        Returns
        -------
        action : string
            Name of action related to writing a checkpoint.
        """
        return action_write_iteration_checkpoint()

    def action_read_checkpoint(self):
        """
        Get name of action to convey to preCICE that a checkpoint has been read and the state of the system has been
        restored to that checkpoint.

        Returns
        -------
        action : string
            Name of action related to reading a checkpoint.
        """
        return action_read_iteration_checkpoint()
