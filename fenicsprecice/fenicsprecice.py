"""
FEniCS - preCICE Adapter. API to help users couple FEniCS with other solvers using the preCICE library.
:raise ImportError: if PRECICE_ROOT is not defined
"""
import numpy as np
from .config import Config
import logging
import precice
from .adapter_core import FunctionType, determine_function_type, convert_fenics_to_precice, get_fenics_vertices, \
    get_owned_vertices, get_unowned_vertices, get_coupling_boundary_edges, get_forces_as_point_sources, \
    get_communication_map, communicate_shared_vertices, CouplingMode, Vertices, VertexType, filter_point_sources
from .expression_core import SegregatedRBFInterpolationExpression, EmptyExpression
from .solverstate import SolverState
from fenics import Function, FunctionSpace
from mpi4py import MPI
import copy

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
    NOTE: dolfin.PointSource use only works in serial
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

        self._interface = precice.Interface(self._config.get_participant_name(), self._config.get_config_file_name(),
                                            self._comm.Get_rank(), self._comm.Get_size())

        # FEniCS related quantities
        self._read_function_space = None  # initialized later
        self._write_function_space = None  # initialized later
        self._dofmap = None  # initialized later using function space provided by user

        # coupling mesh related quantities
        self._owned_vertices = Vertices(VertexType.OWNED)
        self._unowned_vertices = Vertices(VertexType.UNOWNED)
        self._fenics_vertices = Vertices(VertexType.FENICS)
        self._precice_vertex_ids = None  # initialized later

        # read data related quantities (read data is read from preCICE and applied in FEniCS)
        self._read_function_type = None  # stores whether read function is scalar or vector valued
        self._write_function_type = None  # stores whether write function is scalar or vector valued

        # write data related quantities (write data is written to preCICE)
        self._write_function_type = None  # stores whether read function is scalar or vector valued

        # Interpolation strategy
        self._my_expression = SegregatedRBFInterpolationExpression

        # Solver state used by the Adapter internally to handle checkpointing
        self._checkpoint = None

        # Dirichlet boundary for FSI Simulations
        self._Dirichlet_Boundary = None  # stores a dirichlet boundary (if provided)

        # Necessary bools for enforcing proper control flow / warnings to user
        self._first_advance_done = False

        # Parallel communication
        self._send_map = None
        self._recv_map = None
        self._empty_rank = True

        # Determine type of coupling in initialization
        self._coupling_type = None

        # Problem dimension in FEniCS
        self._fenics_dims = None

    def _is_parallel(self):
        """
        Internal function to identify if the adapter is being used for parallel computations

        Returns
        -------
        bool : bool
            True if parallel initialization
        """
        return self._comm.Get_size() > 1

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
                # element information must be provided, else DOLFIN assumes scalar function
                coupling_expression = self._my_expression(element=self._read_function_space.ufl_element())
            except (TypeError, KeyError):  # works with dolfin 2017.2.0
                coupling_expression = self._my_expression(element=self._read_function_space.ufl_element(), degree=0)
        else:
            try:  # works with dolfin 1.6.0
                # element information must be provided, else DOLFIN assumes scalar function
                coupling_expression = EmptyExpression(element=self._read_function_space.ufl_element())
            except (TypeError, KeyError):  # works with dolfin 2017.2.0
                coupling_expression = EmptyExpression(element=self._read_function_space.ufl_element(), degree=0)
            if self._read_function_type == FunctionType.SCALAR:
                # todo: try to find a solution where we don't have to access the private
                # member coupling_expression._vals
                coupling_expression._vals = np.empty(shape=0)
            elif self._read_function_type == FunctionType.VECTOR:
                # todo: try to find a solution where we don't have to access the private
                # member coupling_expression._vals
                coupling_expression._vals = np.empty(shape=(0, 0))

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

    def get_point_sources(self, data):
        """
        Update values of at points by defining a point source load using data.

        Parameters
        ----------
        data : dict_like
            The coupling data. A dictionary containing nodal data with vertex coordinates as key and associated data as
            value.

        Returns
        -------
        x_forces : list
            List containing X component of forces with reference to respective point sources on the coupling subdomain.
        y_forces : list
            List containing Y component of forces with reference to respective point sources on the coupling subdomain.
        """
        assert (self._read_function_type is FunctionType.VECTOR), \
            "PointSources only supported for vector valued read data."

        assert (not self._is_parallel()), "get_point_sources function only works in serial."

        return get_forces_as_point_sources(self._Dirichlet_Boundary, self._read_function_space, data)

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
        assert (self._coupling_type is CouplingMode.UNI_DIRECTIONAL_READ_COUPLING or
                CouplingMode.BI_DIRECTIONAL_COUPLING)

        read_data_id = self._interface.get_data_id(self._config.get_read_data_name(),
                                                   self._interface.get_mesh_id(self._config.get_coupling_mesh_name()))

        read_data = None

        if self._empty_rank:
            assert (self._is_parallel()), "having participants without coupling mesh nodes is only valid for parallel runs"

        if not self._empty_rank:
            if self._read_function_type is FunctionType.SCALAR:
                read_data = self._interface.read_block_scalar_data(read_data_id, self._precice_vertex_ids)
            elif self._read_function_type is FunctionType.VECTOR:
                read_data = self._interface.read_block_vector_data(read_data_id, self._precice_vertex_ids)

            read_data = {tuple(key): value for key, value in zip(self._owned_vertices.get_coordinates(), read_data)}
            read_data = communicate_shared_vertices(
                self._comm, self._fenics_vertices, self._send_map, self._recv_map, read_data)
        else:  # if there are no vertices, we return empty data
            read_data = None

        return copy.deepcopy(read_data)

    def write_data(self, write_function):
        """
        Writes data to preCICE. Depending on the dimensions of the simulation (2D-3D Coupling, 2D-2D coupling or
        Scalar/Vector write function) write_data is first converted into a format needed for preCICE.

        Parameters
        ----------
        write_function : Object of class dolfin.functions.function.Function
            A FEniCS function consisting of the data which this participant will write to preCICE in every time step.
        """

        assert (self._coupling_type is CouplingMode.UNI_DIRECTIONAL_WRITE_COUPLING or
                CouplingMode.BI_DIRECTIONAL_COUPLING)

        w_func = write_function.copy()
        # making sure that the FEniCS function provided by the user is not directly accessed by the Adapter
        assert (w_func != write_function)

        # Check that the function provided lives on the same function space provided during initialization
        assert (self._write_function_type == determine_function_type(w_func))
        assert (write_function.function_space() == self._write_function_space)

        write_data_id = self._interface.get_data_id(self._config.get_write_data_name(),
                                                    self._interface.get_mesh_id(self._config.get_coupling_mesh_name()))

        if self._empty_rank:
            assert (self._is_parallel()), "having participants without coupling mesh nodes is only valid for parallel runs"

        write_function_type = determine_function_type(write_function)
        assert (write_function_type in list(FunctionType))
        write_data = convert_fenics_to_precice(write_function, self._owned_vertices.get_local_ids())
        if write_function_type is FunctionType.SCALAR:
            assert (write_function.function_space().num_sub_spaces() == 0)
            self._interface.write_block_scalar_data(write_data_id, self._precice_vertex_ids, write_data)
        elif write_function_type is FunctionType.VECTOR:
            assert (write_function.function_space().num_sub_spaces() > 0)
            self._interface.write_block_vector_data(write_data_id, self._precice_vertex_ids, write_data)
        else:
            raise Exception("write_function provided is neither VECTOR nor SCALAR type")

    def initialize(self, coupling_subdomain, read_function_space=None, write_object=None, fixed_boundary=None):
        """
        Initializes the coupling and sets up the mesh where coupling happens in preCICE.

        Parameters
        ----------
        coupling_subdomain : Object of class dolfin.cpp.mesh.SubDomain
            SubDomain of mesh which is the physical coupling boundary.
        read_function_space : Object of class dolfin.functions.functionspace.FunctionSpace
            Function space on which the read function lives. If not provided then the adapter assumes that this
            participant is a write-only participant.
        write_object : Object of class dolfin.functions.functionspace.FunctionSpace / dolfin.functions.function.Function
            Function space on which the write function lives or FEniCS function related to the quantity to be written
            by FEniCS during each coupling iteration. If not provided then the adapter assumes that this participant is
            a read-only participant.
        fixed_boundary : Object of class dolfin.fem.bcs.AutoSubDomain
            SubDomain consisting of a fixed boundary of the mesh. For example in FSI cases usually the solid body
            is fixed at one end (fixed end of a flexible beam).

        Returns
        -------
        dt : double
            Recommended time step value from preCICE.
        """

        write_function_space, write_function = None, None
        if isinstance(write_object, Function):  # precice.initialize_data() will be called using this Function
            write_function_space = write_object.function_space()
            write_function = write_object
        elif isinstance(write_object, FunctionSpace):  # preCICE will use default zero values for initialization.
            write_function_space = write_object
            write_function = None
        elif write_object is None:
            pass
        else:
            raise Exception("Given write object is neither of type dolfin.functions.function.Function or "
                            "dolfin.functions.functionspace.FunctionSpace")

        if isinstance(read_function_space, FunctionSpace):
            pass
        elif read_function_space is None:
            pass
        else:
            raise Exception("Given read_function_space is not of type dolfin.functions.functionspace.FunctionSpace")

        if read_function_space is None and write_function_space:
            self._coupling_type = CouplingMode.UNI_DIRECTIONAL_WRITE_COUPLING
            assert (self._config.get_write_data_name())
            print("Participant {} is write-only participant".format(self._config.get_participant_name()))
            function_space = write_function_space
        elif read_function_space and write_function_space is None:
            self._coupling_type = CouplingMode.UNI_DIRECTIONAL_READ_COUPLING
            assert (self._config.get_read_data_name())
            print("Participant {} is read-only participant".format(self._config.get_participant_name()))
            function_space = read_function_space
        elif read_function_space and write_function_space:
            self._coupling_type = CouplingMode.BI_DIRECTIONAL_COUPLING
            assert (self._config.get_read_data_name() and self._config.get_write_data_name())
            function_space = read_function_space
        elif read_function_space is None and write_function_space is None:
            raise Exception("Neither read_function_space nor write_function_space is provided. Please provide a "
                            "write_object if this participant is used in one-way coupling and only writes data. "
                            "Please provide a read_function_space if this participant is used in one-way coupling and "
                            "only reads data. If two-way coupling is implemented then both read_function_space"
                            " and write_object need to be provided.")
        else:
            raise Exception("Incorrect read and write function space combination provided. Please check input "
                            "parameters in initialization")

        if self._coupling_type is CouplingMode.UNI_DIRECTIONAL_READ_COUPLING or \
                self._coupling_type is CouplingMode.BI_DIRECTIONAL_COUPLING:
            self._read_function_type = determine_function_type(read_function_space)
            self._read_function_space = read_function_space

        if self._coupling_type is CouplingMode.UNI_DIRECTIONAL_WRITE_COUPLING or \
                self._coupling_type is CouplingMode.BI_DIRECTIONAL_COUPLING:
            # Ensure that function spaces of read and write functions are defined using the same mesh
            self._write_function_type = determine_function_type(write_function_space)
            self._write_function_space = write_function_space

        coords = function_space.tabulate_dof_coordinates()
        _, self._fenics_dims = coords.shape

        # Ensure that function spaces of read and write functions use the same mesh
        if self._coupling_type is CouplingMode.BI_DIRECTIONAL_COUPLING:
            assert (self._read_function_space.mesh() is write_function_space.mesh()
                    ), "read_function_space and write_object need to be defined using the same mesh"

        if fixed_boundary:
            self._Dirichlet_Boundary = fixed_boundary

        if self._fenics_dims != 2:
            raise Exception("Currently the fenics-adapter only supports 2D cases")

        if self._fenics_dims != self._interface.get_dimensions():
            raise Exception("Dimension of preCICE setup and FEniCS do not match")

        # Set vertices on the coupling subdomain for this rank
        lids, gids, coords = get_fenics_vertices(function_space, coupling_subdomain, self._fenics_dims)
        self._fenics_vertices.set_local_ids(lids)
        self._fenics_vertices.set_global_ids(gids)
        self._fenics_vertices.set_coordinates(coords)

        if self._is_parallel():
            lids, gids, coords = get_owned_vertices(function_space, coupling_subdomain, self._fenics_dims)
            self._owned_vertices.set_local_ids(lids)
            self._owned_vertices.set_global_ids(gids)
            self._owned_vertices.set_coordinates(coords)

            gids = get_unowned_vertices(function_space, coupling_subdomain, self._fenics_dims)
            self._unowned_vertices.set_global_ids(gids)
        else:
            # For serial execution, owned vertices are identical to fenics vertices
            self._owned_vertices.set_local_ids(lids)
            self._owned_vertices.set_global_ids(gids)
            self._owned_vertices.set_coordinates(coords)

        # Set up mesh in preCICE
        if self._fenics_vertices.get_global_ids().size > 0:
            self._empty_rank = False
        else:
            print("Rank {} has no part of coupling boundary.".format(self._comm.Get_rank()))

        # Define mesh in preCICE
        self._precice_vertex_ids = self._interface.set_mesh_vertices(self._interface.get_mesh_id(
            self._config.get_coupling_mesh_name()), self._owned_vertices.get_coordinates())

        if self._coupling_type is CouplingMode.UNI_DIRECTIONAL_READ_COUPLING or \
                self._coupling_type is CouplingMode.BI_DIRECTIONAL_COUPLING:
            # Determine shared vertices with neighbouring processes and get dictionaries for communication
            self._send_map, self._recv_map = get_communication_map(self._comm, self._read_function_space,
                                                                   self._owned_vertices,
                                                                   self._unowned_vertices)

        # Check for double boundary points
        if fixed_boundary:
            # create empty data for the sake of searching for duplicate points
            point_data = {tuple(key): None for key in self._owned_vertices.get_coordinates()}
            _ = filter_point_sources(point_data, fixed_boundary, warn_duplicate=True)

        # Set mesh edges in preCICE to allow nearest-projection mapping
        # Define a mapping between coupling vertices and their IDs in preCICE
        id_mapping = {key: value for key, value in zip(self._owned_vertices.get_global_ids(), self._precice_vertex_ids)}

        edge_vertex_ids1, edge_vertex_ids2 = get_coupling_boundary_edges(function_space, coupling_subdomain,
                                                                         self._owned_vertices.get_global_ids(),
                                                                         id_mapping)

        for i in range(len(edge_vertex_ids1)):
            assert (edge_vertex_ids1[i] != edge_vertex_ids2[i])
            self._interface.set_mesh_edge(self._interface.get_mesh_id(self._config.get_coupling_mesh_name()),
                                          edge_vertex_ids1[i], edge_vertex_ids2[i])

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
        Finalizes the coupling via preCICE and the adapter. To be called at the end of the simulation.

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
