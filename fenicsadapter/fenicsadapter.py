"""
Adapter module to provide an API to simulate partitioned coupled problems with FEniCS as a coupling participant
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
    This adapter class provides an interface to the preCICE coupling library for setting up a coupling case which has
    FEniCS as a participant for 2D problems.
    The user can create and manage a dolfin.UserExpression and/or dolfin.PointSource at the coupling boundary.
    Reading data from preCICE and writing data to preCICE is also managed via functions of this class.
    If the user wants to perform implicit coupling then a steering mechanism for checkpointing is also provided.

    For more information on setting up a coupling case using dolfin.UserExpression at the coupling boundary please have
    a look at this tutorial:
    https://github.com/precice/tutorials/tree/master/HT/partitioned-heat/fenics-fenics

    For more information on setting up a coupling case using dolfin.PointSource at the coupling boundary please have a
    look at this tutorial:
    https://github.com/precice/tutorials/tree/master/FSI/flap_perp/OpenFOAM-FEniCS
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

        self._interface = precice.Interface(self._config.get_participant_name(), self._config.get_config_file_name(), 0, 1)

        # FEniCS related quantities
        self._fenics_dimensions = None
        self._function_space = None  # initialized later

        # coupling mesh related quantities
        self._coupling_mesh_vertices = None  # initialized later
        self._vertex_ids = None  # initialized later

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
            warn("No valid interpolation strategy entered. It is assumed that the user does not wish to use FEniCS Expressions on the coupling boundary.")

        # Solver state used by the Adapter internally to handle checkpointing
        self._checkpoint = None

        # Dirichlet boundary for FSI Simulations
        self._Dirichlet_Boundary = None  # stores a dirichlet boundary (if provided)

        # Necessary bools for enforcing proper control flow / warnings to user
        self._first_advance_done = False
        self._apply_2d_3d_coupling = False
        self._non_standard_initialization = False

    def create_coupling_expression(self, data=None):
        """
        Creates a FEniCS Expression in the form of an object of class GeneralInterpolationExpression or
        ExactInterpolationExpression. The adapter will hold this object till the coupling is on going.

        Parameters
        ----------
        data : array_like
            The coupling data. A numpy array [N x D] where N = number of vertices and D = dimensions of the data, i.e.
            for scalar valued data D = 1 and for vector valued data D = dimension of problem.
            If no data is provided then a numpy array of shape (N, D) with zero values is used.

        Returns
        -------
        coupling_expression : Object of class dolfin.functions.expression.Expression
            Reference to object of class GeneralInterpolationExpression or ExactInterpolationExpression.
        """
        if self._non_standard_initialization and data is None:
            warn("initialize_data() was called but the data was not supplied for creating the coupling expression")

        try:  # works with dolfin 1.6.0
            # element information must be provided, else DOLFIN assumes scalar function
            coupling_expression = self._my_expression(element=self._function_space.ufl_element())
        except (TypeError, KeyError):  # works with dolfin 2017.2.0
            coupling_expression = self._my_expression(element=self._function_space.ufl_element(), degree=0)

        n_vertices, _ = self._coupling_mesh_vertices.shape

        if data is None:
            # standard initialization with zero valued data
            data = np.zeros((n_vertices, self._fenics_dimensions))

        self.update_coupling_expression(coupling_expression, data)

        return coupling_expression

    def update_coupling_expression(self, coupling_expression, data):
        """
        Updates the given FEniCS Expression using provided data. The boundary data is updated.
        User needs to explicitly call this function in each time step.

        Parameters
        ----------
        coupling_expression : Object of class dolfin.functions.expression.Expression
            Reference to object of class GeneralInterpolationExpression or ExactInterpolationExpression.
        data : array_like
            The coupling data. A numpy array [N x D] where N = number of vertices and D = dimensions of the data, i.e.
            for scalar valued data D = 1 and for vector valued data D = dimension of problem.
        """
        coupling_expression.update_boundary_data(data, self._coupling_mesh_vertices[:, 0], self._coupling_mesh_vertices[:, 1])

    def create_point_sources(self, fixed_boundary, data=None):
        """
        Create point sources with reference to fixed boundary in a FSI simulation.

        Parameters
        ----------
        fixed_boundary : Object of class dolfin.fem.bcs.AutoSubDomain
            SubDomain consisting of a fixed boundary condition. For example in FSI cases usually the solid body
            is fixed at one end (fixed end of a flexible beam).
        data : array_like
            The coupling data. A numpy array [N x D] where N = number of vertices and D = dimensions of the data, i.e.
            for scalar valued data D = 1 and for vector valued data D = dimension of problem.
            If no data is provided then a numpy array of shape (N, D) with zero values is used.

        Returns
        -------
        x_forces : list
            List containing X component of forces with reference to respective point sources on the coupling interface.
        y_forces : list
            List containing Y component of forces with reference to respective point sources on the coupling interface.
        """
        if self._non_standard_initialization and data is None:
            warn("initialize_data() was called but the data was not supplied for creating the point sources")

        n_vertices, _ = self._coupling_mesh_vertices.shape

        if data is None:
            data = np.zeros((n_vertices, self._fenics_dimensions))

        self._Dirichlet_Boundary = fixed_boundary
        return self.update_point_sources(data)

    def update_point_sources(self, data):
        """
        Update values of point sources using data.

        Parameters
        ----------
        data : array_like
            The coupling data. A numpy array [N x D] where N = number of vertices and D = dimensions of the data, i.e.
            for scalar valued data D = 1 and for vector valued data D = dimension of problem.

        Returns
        -------
        x_forces : list
            List containing X component of forces with reference to respective point sources on the coupling interface.
        y_forces : list
            List containing Y component of forces with reference to respective point sources on the coupling interface.
        """
        return get_forces_as_point_sources(self._Dirichlet_Boundary, self._function_space, self._coupling_mesh_vertices,
                                           data)

    def read_data(self):
        """
        Read data from preCICE. Data is generated depending on the type of the read function (Scalar or Vector).
        For a scalar read function the data is a numpy array with shape (N) where N = number of coupling vertices
        For a vector read function the data is a numpy array with shape (N, D) where
        N = number of coupling vertices and D = dimensions of FEniCS setup

        Note: For quasi 2D-3D coupled simulation (FEniCS participant is 2D) the Z-component of the data is deleted.

        Returns
        -------
        read_data : array_like
            Numpy array containing the read data.
        """
        assert (self._read_function_type in list(FunctionType))

        read_data_id = self._interface.get_data_id(self._config.get_read_data_name(),
                                                   self._interface.get_mesh_id(self._config.get_coupling_mesh_name()))

        if self._read_function_type is FunctionType.SCALAR:
            read_data = self._interface.read_block_scalar_data(read_data_id, self._vertex_ids)
        elif self._read_function_type is FunctionType.VECTOR:
            if self._fenics_dimensions == self._interface.get_dimensions():
                read_data = self._interface.read_block_vector_data(read_data_id, self._vertex_ids)
            elif self._apply_2d_3d_coupling:
                precice_read_data = self._interface.read_block_vector_data(read_data_id, self._vertex_ids)
                n_vertices, dims = precice_read_data.shape
                read_data = np.zeros((n_vertices, dims-1))
                read_data[:, 0] = precice_read_data[:, 0]
                read_data[:, 1] = precice_read_data[:, 1]
                # z is the dead direction so the data is not transferred to read_data array
                assert (np.sum(np.abs(precice_read_data[:, 2])) < 1e-10)
            else:
                raise Exception("Dimensions do not match.")
        else:
            raise Exception("Rank of function space is neither 0 nor 1")

        return read_data

    def write_data(self, write_function):
        """
        Writes data to preCICE. Depending on the dimensions of the simulation (2D-3D Coupling, 2D-2D coupling or
        Scalar/Vector write function) write_data is first converted into a format needed for preCICE.

        Parameters
        ----------
        write_function : Object of class dolfin.functions.function.Function
            A FEniCS function consisting of the data which this participant will write to preCICE in every time step.
        """
        write_function_type = determine_function_type(write_function)
        assert (write_function_type in list(FunctionType))

        write_data = convert_fenics_to_precice(write_function, self._coupling_mesh_vertices)

        write_data_id = self._interface.get_data_id(self._config.get_write_data_name(),
                                                    self._interface.get_mesh_id(self._config.get_coupling_mesh_name()))

        n_vertices, _ = self._coupling_mesh_vertices.shape

        if write_function_type is FunctionType.SCALAR:
            self._interface.write_block_scalar_data(write_data_id, self._vertex_ids, write_data)
        elif write_function_type is FunctionType.VECTOR:
            if self._apply_2d_3d_coupling:
                # in 2d-3d coupling z dimension is set to zero
                precice_write_data = np.column_stack((write_data[:, 0], write_data[:, 1], np.zeros(n_vertices)))
                assert (precice_write_data.shape[0] == n_vertices and
                        precice_write_data.shape[1] == self._interface.get_dimensions())
                self._interface.write_block_vector_data(write_data_id, self._vertex_ids, precice_write_data)
            elif self._fenics_dimensions == self._interface.get_dimensions():
                self._interface.write_block_vector_data(write_data_id, self._vertex_ids, write_data)
            else:
                raise Exception("Dimensions of FEniCS problem and coupling configuration do not match.")
        else:
            raise Exception("write_function provided is neither VECTOR nor SCALAR type")

    def initialize(self, coupling_subdomain, mesh, read_function, function_space, dimensions=2):
        """
        Initializes the coupling interface and sets up the mesh in preCICE.

        Parameters
        ----------
        coupling_subdomain : Object of class dolfin.cpp.mesh.SubDomain
            SubDomain of mesh which is the physical coupling boundary.
        mesh : Object of class dolfin.cpp.mesh.Mesh
            SubDomain of mesh of the complete region.
        read_function : Object of class dolfin.functions.function.Function
            FEniCS function related to the quantity to be read by FEniCS during each coupling iteration.
        function_space : Object of class dolfin.functions.functionspace.FunctionSpace
            Function space on which the finite element formulation of the problem lives.
        dimensions : int
            Dimensions of the problem as defined in FEniCS.

        Returns
        -------
        dt : double
            Recommended time step value from preCICE.
        """

        if not read_function:
            raise Exception("Please provide a read_function. Explicit coupling is not yet supported for the FEniCS adapter.")

        self._fenics_dimensions = dimensions

        if dimensions != self._interface.get_dimensions():
            logger.warning("fenics_dimension = {} and precice_dimension = {} do not match!".format(
                dimensions, self._interface.get_dimensions()))

            if dimensions == 2 and self._interface.get_dimensions() == 3:
                logger.warning("2D-3D coupling will be applied. Z coordinates of all nodes will be set to zero.")
                self._apply_2d_3d_coupling = True
            else:
                raise Exception("fenics_dimension = {}, precice_dimension = {}. "
                                "No proper treatment for dimensional mismatch is implemented. Aborting!".format(
                    dimensions, self._interface.get_dimensions()))

        fenics_vertices, self._coupling_mesh_vertices = get_coupling_boundary_vertices(
            mesh, coupling_subdomain, dimensions, self._interface.get_dimensions())

        # Set up mesh in preCICE
        self._vertex_ids = self._interface.set_mesh_vertices(self._interface.get_mesh_id(
            self._config.get_coupling_mesh_name()), self._coupling_mesh_vertices)

        # Define a mapping between coupling vertices and their IDs in preCICE
        id_mapping = dict()
        n_vertices, _ = self._coupling_mesh_vertices.shape
        for i in range(n_vertices):
            id_mapping[fenics_vertices[i].global_index()] = self._vertex_ids[i]
        edge_vertex_ids1, edge_vertex_ids2 = get_coupling_boundary_edges(mesh, coupling_subdomain, id_mapping)

        # Set mesh edges in preCICE to allow nearest-projection mapping
        for i in range(len(edge_vertex_ids1)):
            assert (edge_vertex_ids1[i] != edge_vertex_ids2[i])
            self._interface.set_mesh_edge(self._interface.get_mesh_id(self._config.get_coupling_mesh_name()),
                                          edge_vertex_ids1[i], edge_vertex_ids2[i])

        # Set read functionality parameters
        self._read_function_type = determine_function_type(read_function)
        self._function_space = function_space

        return self._interface.initialize()

    def initialize_data(self, write_function=None):
        """
        Set non-standard values as initial conditions to coupling problem

        Parameters
        ----------
        write_function : Object of class dolfin.functions.function.Function
            FEniCS function related to the quantity to be written by FEniCS during each coupling iteration.

        Returns
        -------
        read_data = array_like
            Numpy array containing the read data.
        """
        self._non_standard_initialization = True

        if write_function:
            if self._interface.is_action_required(action_write_initial_data()):
                self.write_data(write_function)
                self._interface.mark_action_fulfilled(action_write_initial_data())

        self._interface.initialize_data()

        read_data = None
        if self._interface.is_read_data_available():
            read_data = self.read_data()

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
        assert (my_u != user_u)  # wrt to pointer, make sure the direct function reference is not stored
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
        assert (not self.is_time_window_complete())
        logger.debug("Restore solver state")
        self._interface.mark_action_fulfilled(self.action_read_checkpoint())
        return self._checkpoint.get_state()

    def advance(self, dt):
        """
        Advances coupling in preCICE.

        Parameters
        ----------
        dt : double
            Length of timestep used by the solver.

        Notes
        -----
        Refer advance() in https://github.com/precice/python-bindings/blob/develop/precice.pyx

        Returns
        -------
        max_dt : double
            Maximum length of timestep to be computed by solver.
        """
        self._first_advance_done = True
        max_dt = self._interface.advance(dt)
        return max_dt

    def finalize(self):
        """
        Completes the the coupling interface execution. To be called at the end of the simulation.

        Notes
        -----
        Refer finalize() in https://github.com/precice/python-bindings/blob/develop/precice.pyx
        """
        self._interface.finalize()

    def get_participant_name(self):
        """
        Returns
        -------
        participant_name : string
            Name of the participant.
        """
        return self._config.get_participant_name()

    def is_coupling_ongoing(self):
        """
        Checks if the coupled simulation is still ongoing.

        Notes
        -----
        Refer is_coupling_ongoing() in https://github.com/precice/python-bindings/blob/develop/precice.pyx

        Returns
        -------
        tag : bool
            True if coupling is still going on and False if coupling has finished.
        """
        return self._interface.is_coupling_ongoing()

    def is_time_window_complete(self):
        """
        Tag to check if implicit iteration has converged.

        Notes
        -----
        Refer is_time_window_complete() in https://github.com/precice/python-bindings/blob/develop/precice.pyx

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

        Notes
        -----
        Refer is_action_required(action) in https://github.com/precice/python-bindings/blob/develop/precice.pyx

        Returns
        -------
        tag : bool
            True if action is required and False if action is not required.
        """
        return self._interface.is_action_required(action)

    def action_write_checkpoint(self):
        """
        Get name of action to convey to preCICE that a checkpoint has been written.

        Notes
        -----
        Refer action_write_iteration_checkpoint() in https://github.com/precice/python-bindings/blob/develop/precice.pyx

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

        Notes
        -----
        Refer action_read_iteration_checkpoint() in https://github.com/precice/python-bindings/blob/develop/precice.pyx

        Returns
        -------
        action : string
            Name of action related to reading a checkpoint.
        """
        return action_read_iteration_checkpoint()
