"""
Adapter to FEniCS solver handles CustomExpression and initialization of the FEniCS adapter.
:raise ImportError: if PRECICE_ROOT is not defined
"""
import numpy as np
from .config import Config
import logging
import precice
from .adapter_core import FunctionType, determine_function_type, convert_fenics_to_precice, \
    get_fenics_coupling_boundary_vertices, get_owned_coupling_boundary_vertices, get_unowned_coupling_boundary_vertices, \
    get_coupling_boundary_edges, get_forces_as_point_sources, determine_shared_vertices, communicate_shared_vertices
from .expression_core import SegregatedRBFInterpolationExpression, EmptyExpression
from .solverstate import SolverState
from mpi4py import MPI
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

        # Setup up MPI communicator on mpi4py
        self._comm = MPI.COMM_WORLD
        self._rank = self._comm.Get_rank()
        self._size = self._comm.Get_size()

        self._interface = precice.Interface(self._config.get_participant_name(), self._config.get_config_file_name(),
                                            self._rank, self._size)

        # FEniCS related quantities
        self._read_function_space = None  # initialized later
        self._write_function_space = None  # initialized later
        self._dofmap = None  # initialized later using function space provided by user
        self._fenics_lids = None
        self._fenics_gids = None
        self._fenics_coords = None
        self._coupling_subdomain = None

        # coupling mesh related quantities
        self._owned_gids = None  # initialized later
        self._owned_lids = None  # initialized later
        self._owned_coords = None  # initialized later
        self._unowned_gids = None
        self._unowned_lids = None
        self._vertex_ids = None  # initialized later

        # read data related quantities (read data is read by use to FEniCS from preCICE)
        self._read_function_type = None  # stores whether read function is scalar or vector valued
        self._write_function_type = None  # stores whether write function is scalar or vector valued

        # Interpolation strategy
        self._my_expression = SegregatedRBFInterpolationExpression

        # Solver state used by the Adapter internally to handle checkpointing
        self._checkpoint = None

        # Dirichlet boundary for FSI Simulations
        self._Dirichlet_Boundary = None  # stores a dirichlet boundary (if provided)

        # Necessary bools for enforcing proper control flow / warnings to user
        self._first_advance_done = False

        # Necessary data for parallel computations
        self._to_send_pts = None
        self._to_recv_pts = None
        self._empty_rank = True

    def create_coupling_expression(self):
        """
        Creates a FEniCS Expression in the form of an object of class GeneralInterpolationExpression or
        ExactInterpolationExpression. The adapter will hold this object till the coupling is on going.

        Returns
        -------
        coupling_expression : Object of class dolfin.functions.expression.Expression
            Reference to object of class GeneralInterpolationExpression or ExactInterpolationExpression.
        """

        if not (self._read_function_type is FunctionType.SCALAR or self._read_function_type is FunctionType.VECTOR):
            raise Exception("No valid read_function is provided in initialization. Cannot create coupling expression")

        if not self._empty_rank:
            try:  # works with dolfin 1.6.0
                coupling_expression = self._my_expression(
                    element=self._read_function_space.ufl_element())  # element information must be provided, else DOLFIN assumes scalar function
            except (TypeError, KeyError):  # works with dolfin 2017.2.0
                coupling_expression = self._my_expression(element=self._read_function_space.ufl_element(), degree=0)
        else:
            try:  # works with dolfin 1.6.0
                coupling_expression = EmptyExpression(
                    element=self._read_function_space.ufl_element())  # element information must be provided, else DOLFIN assumes scalar function
            except (TypeError, KeyError):  # works with dolfin 2017.2.0
                coupling_expression = EmptyExpression(element=self._read_function_space.ufl_element(), degree=0)
            if self._read_function_type == FunctionType.SCALAR:
                coupling_expression._vals = np.empty(
                    shape=0)  # todo: try to find a solution where we don't have to access the private member coupling_expression._vals
            elif self._read_function_type == FunctionType.VECTOR:
                coupling_expression._vals = np.empty(shape=(0,
                                                            0))  # todo: try to find a solution where we don't have to access the private member coupling_expression._vals

        coupling_expression.set_function_type(self._read_function_type)

        return coupling_expression

    def update_coupling_expression(self, coupling_expression, data):
        """
        Updates the given FEniCS Expression using provided data. The boundary data is updated.
        User needs to explicitly call this function in each time step.

        Parameters
        ----------
        coupling_expression : Object of class dolfin.functions.expression.Expression
            Reference to object of class GeneralInterpolationExpression or ExactInterpolationExpression.
        data : dict_like
            The coupling data. A dictionary containing nodal data with vertex coordinates as key and associated data as
            value.
        """
        if not self._empty_rank:
            vertices = np.array(list(data.keys()))
            nodal_data = np.array(list(data.values()))
            coupling_expression.update_boundary_data(nodal_data, vertices[:, 0], vertices[:, 1])
        else:
            print("Rank {}: No update for Expression as this rank has no vertices".format(self._rank))

    def get_point_sources(self, data):
        """
        Update values of point sources using data.

        Parameters
        ----------
        data : dict_like
            The coupling data. A dictionary containing nodal data with vertex coordinates as key and associated data as
            value.

        Returns
        -------
        x_forces : list
            List containing X component of forces with reference to respective point sources on the coupling interface.
        y_forces : list
            List containing Y component of forces with reference to respective point sources on the coupling interface.
        """
        assert (self._read_function_type is FunctionType.VECTOR), \
            "PointSources only supported for vector valued read data."

        vertices = np.array(list(data.keys()))
        nodal_data = np.array(list(data.values()))

        return get_forces_as_point_sources(self._Dirichlet_Boundary, self._read_function_space, vertices, nodal_data)

    def read_data(self):
        """
        Read data from preCICE. Data is generated depending on the type of the read function (Scalar or Vector).
        For a scalar read function the data is a numpy array with shape (N) where N = number of coupling vertices
        For a vector read function the data is a numpy array with shape (N, D) where
        N = number of coupling vertices and D = dimensions of FEniCS setup

        Note: For quasi 2D-3D coupled simulation (FEniCS participant is 2D) the Z-component of the data and vertices
        is deleted.

        Returns
        -------
        data : dict_like
            The coupling data. A dictionary containing nodal data with vertex coordinates as key and associated data as
            value.
        """
        read_data_id = self._interface.get_data_id(self._config.get_read_data_name(),
                                                   self._interface.get_mesh_id(self._config.get_coupling_mesh_name()))

        read_data = None

        if self._empty_rank:
            assert (self._size > 1)  # having participants without coupling mesh nodes is only valid for parallel runs

        if not self._empty_rank:
            if self._read_function_type is FunctionType.SCALAR:
                read_data = self._interface.read_block_scalar_data(read_data_id, self._vertex_ids)
            elif self._read_function_type is FunctionType.VECTOR:
                read_data = self._interface.read_block_vector_data(read_data_id, self._vertex_ids)

            owned_read_data = {tuple(key): value for key, value in zip(self._owned_coords, read_data)}
            updated_data = communicate_shared_vertices(self._comm, self._rank, self._fenics_gids,
                                                       self._owned_coords, self._fenics_coords, owned_read_data,
                                                       self._to_send_pts, self._to_recv_pts)
        else:  # if there are no vertices, we return empty data
            updated_data = None

        return updated_data

    def write_data(self, write_function):
        """
        Writes data to preCICE. Depending on the dimensions of the simulation (2D-3D Coupling, 2D-2D coupling or
        Scalar/Vector write function) write_data is first converted into a format needed for preCICE.

        Parameters
        ----------
        write_function : Object of class dolfin.functions.function.Function
            A FEniCS function consisting of the data which this participant will write to preCICE in every time step.
        """
        w_func = write_function.copy()
        # making sure that the FEniCS function provided by the user is not directly accessed by the Adapter
        assert (w_func != write_function)

        # Check that the function provided lives on the same function space provided during initialization
        assert (self._write_function_type == determine_function_type(w_func))
        assert (write_function.function_space() == self._write_function_space)

        write_data_id = self._interface.get_data_id(self._config.get_write_data_name(),
                                                    self._interface.get_mesh_id(self._config.get_coupling_mesh_name()))

        if self._empty_rank:
            assert (self._size > 1)  # having participants without coupling mesh nodes is only valid for parallel runs

        write_function_type = determine_function_type(write_function)
        assert (write_function_type in list(FunctionType))
        write_data = convert_fenics_to_precice(write_function, self._coupling_subdomain, self._interface.get_dimensions())
        if write_function_type is FunctionType.SCALAR:
            assert (write_function.function_space().num_sub_spaces() == 0)
            self._interface.write_block_scalar_data(write_data_id, self._vertex_ids, write_data)
        elif write_function_type is FunctionType.VECTOR:
            assert (write_function.function_space().num_sub_spaces() > 0)
            self._interface.write_block_vector_data(write_data_id, self._vertex_ids, write_data)
        else:
            raise Exception("write_function provided is neither VECTOR nor SCALAR type")

    def initialize(self, coupling_subdomain, read_function_space=None, write_function_space=None, write_function=None,
                   fixed_boundary=None):
        """
        Initializes the coupling interface and sets up the mesh in preCICE. Allows to initialize data on coupling interface.

        Parameters
        ----------
        coupling_subdomain : Object of class dolfin.cpp.mesh.SubDomain
            SubDomain of mesh which is the physical coupling boundary.
        read_function_space : Object of class dolfin.functions.functionspace.FunctionSpace
            Function space on which the finite element formulation of the problem lives.
        write_function_space : Object of class dolfin.functions.functionspace.FunctionSpace
            Function space on which the finite element formulation of the problem lives.
        write_function : Object of class dolfin.functions.function.Function
            FEniCS function related to the quantity to be written by FEniCS during each coupling iteration.
        fixed_boundary : Object of class dolfin.fem.bcs.AutoSubDomain
            SubDomain consisting of a fixed boundary condition. For example in FSI cases usually the solid body
            is fixed at one end (fixed end of a flexible beam).

        Returns
        -------
        dt : double
            Recommended time step value from preCICE.
        """

        # Set read functionality parameters
        self._read_function_type = determine_function_type(read_function_space)
        self._write_function_type = determine_function_type(write_function_space)
        self._read_function_space = read_function_space
        self._write_function_space = write_function_space
        self._coupling_subdomain = coupling_subdomain
        _, fenics_dimensions = self._read_function_space.tabulate_dof_coordinates().shape

        # Ensure that read and write function spaces are defined using the same mesh
        assert self._read_function_space.mesh() is self._write_function_space.mesh()

        if fixed_boundary:
            self._Dirichlet_Boundary = fixed_boundary

        if fenics_dimensions != 2:
            raise Exception("Currently the fenics-adapter only supports 2D cases")

        if fenics_dimensions != self._interface.get_dimensions():
            raise Exception("Dimension of preCICE setup and FEniCS do not match")

        # Get Global IDs and coordinates of vertices on the coupling interface which are owned by this rank
        self._fenics_gids, self._fenics_lids, self._fenics_coords = \
            get_fenics_coupling_boundary_vertices(self._read_function_space, coupling_subdomain)

        self._owned_gids, self._owned_lids, self._owned_coords = \
            get_owned_coupling_boundary_vertices(self._read_function_space, coupling_subdomain)

        self._unowned_gids, self._unowned_lids = \
            get_unowned_coupling_boundary_vertices(self._read_function_space, coupling_subdomain)

        # Set up mesh in preCICE
        if self._fenics_gids.size > 0:
            self._empty_rank = False
            print("Rank {}: fenics_gids = {}".format(self._rank, self._fenics_gids))
            print("Rank {}: fenics_lids = {}".format(self._rank, self._fenics_lids))
            print("Rank {}: fenics_coords = {}".format(self._rank, self._fenics_coords))

            print("Rank {}: owned_gids = {}".format(self._rank, self._owned_gids))
            print("Rank {}: owned_lids = {}".format(self._rank, self._owned_lids))
            print("Rank {}: owned_coords = {}".format(self._rank, self._owned_coords))

            print("Rank {}: unowned_gids = {}".format(self._rank, self._unowned_gids))
            print("Rank {}: unowned_lids = {}".format(self._rank, self._unowned_lids))
        else:
            print("Rank {} is EMPTY RANK".format(self._rank))

        # Define mesh in preCICE
        self._vertex_ids = self._interface.set_mesh_vertices(self._interface.get_mesh_id(
            self._config.get_coupling_mesh_name()), self._owned_coords)

        # Determine shared vertices with neighbouring processes and get dictionaries for communication
        self._to_send_pts, self._to_recv_pts = determine_shared_vertices(self._comm, self._rank,
                                                                         self._read_function_space,
                                                                         self._owned_gids, self._unowned_gids)

        if self._empty_rank is False:
            print("Rank {}: to_send_pts = {}".format(self._rank, self._to_send_pts))
            print("Rank {}: to_recv_pts = {}".format(self._rank, self._to_recv_pts))

        # # Set mesh edges in preCICE to allow nearest-projection mapping
        # # Define a mapping between coupling vertices and their IDs in preCICE
        # id_mapping = {tuple(key): value for key, value in zip(owned_gids, self._vertex_ids)}
        #
        # edge_vertex_ids1, edge_vertex_ids2 = get_coupling_boundary_edges(function_space, coupling_subdomain,
        # id_mapping)
        # for i in range(len(edge_vertex_ids1)): assert (edge_vertex_ids1[i] != edge_vertex_ids2[i])
        # self._interface.set_mesh_edge(self._interface.get_mesh_id(self._config.get_coupling_mesh_name()),
        # edge_vertex_ids1[i], edge_vertex_ids2[i])

        precice_dt = self._interface.initialize()

        if self._interface.is_action_required(precice.action_write_initial_data()):
            if not write_function:
                raise Exception("Non-standard initialization requires a write_function")
            self.write_data(write_function)
            self._interface.mark_action_fulfilled(precice.action_write_initial_data())

        self._interface.initialize_data()

        return precice_dt

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
        # making sure that the FEniCS function provided by user is not directly accessed by the Adapter
        assert (my_u != user_u)
        self._checkpoint = SolverState(my_u, t, n)
        self._interface.mark_action_fulfilled(self.action_write_iteration_checkpoint())

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
        self._interface.mark_action_fulfilled(self.action_read_iteration_checkpoint())
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
        Completes the coupling interface execution. To be called at the end of the simulation.

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

    def action_write_iteration_checkpoint(self):
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
        return precice.action_write_iteration_checkpoint()

    def action_read_iteration_checkpoint(self):
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
        return precice.action_read_iteration_checkpoint()