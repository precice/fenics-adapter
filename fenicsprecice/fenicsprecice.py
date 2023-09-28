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
    get_communication_map, communicate_shared_vertices, CouplingMode, Vertices, VertexType, filter_point_sources, get_coupling_triangles
from .expression_core import SegregatedRBFInterpolationExpression, EmptyExpression
from .solverstate import SolverState
from fenics import Function, FunctionSpace
from mpi4py import MPI
import copy
import os

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
    https://github.com/precice/tutorials/tree/master/partitioned-heat-conduction/fenics

    For more information on setting up a coupling case using dolfin.PointSource at the coupling boundary please have a
    look at this tutorial:
    https://github.com/precice/tutorials/tree/master/perpendicular-flap/solid-fenics
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
        self._config = Config(os.path.relpath(adapter_config_filename))

        # Setup up MPI communicator on mpi4py
        self._comm = MPI.COMM_WORLD

        self._participant = precice.Participant(
            self._config.get_participant_name(),
            self._config.get_config_file_name(),
            self._comm.Get_rank(),
            self._comm.Get_size())

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
        Creates a FEniCS Expression in the form of an object of class SegregatedRBFInterpolationExpression. The adapter
        will hold this object till the coupling is on going.

        Returns
        -------
        coupling_expression : Object of class dolfin.functions.expression.Expression
            Reference to object of class SegregatedRBFInterpolationExpression.
        """
        assert (self._fenics_dims == 2), "Boundary conditions of Expression objects are only allowed for 2D cases"

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
            Reference to object of class SegregatedRBFInterpolationExpression.
        data : dict_like
            The coupling data. A dictionary containing nodal data with vertex coordinates as key and associated data as
            value.
        """
        assert (self._fenics_dims == 2), "Boundary conditions of Expression objects are only allowed for 2D cases"

        if not self._empty_rank:
            coupling_expression.update_boundary_data(np.array(list(data.values())), np.array(list(data.keys())))

    def get_point_sources(self, data):
        """
        Update values at points by defining a point source load using data.

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

    def read_data(self, dt):
        """
        Read data from preCICE. Data is generated depending on the type of the read function (Scalar or Vector).
        For a scalar read function the data is a numpy array with shape (N) where N = number of coupling vertices
        For a vector read function the data is a numpy array with shape (N, D) where
        N = number of coupling vertices and D = dimensions of FEniCS setup

        Returns
        -------
        data : dict_like
            The coupling data. A dictionary containing nodal data with vertex coordinates as key and associated data as
            value.
        """
        assert (self._coupling_type is CouplingMode.UNI_DIRECTIONAL_READ_COUPLING or
                CouplingMode.BI_DIRECTIONAL_COUPLING)

        read_data = None

        if self._empty_rank:
            assert (self._is_parallel()
                    ), "having participants without coupling mesh nodes is only valid for parallel runs"

        if not self._empty_rank:
            read_data = self._participant.read_data(
                self._config.get_coupling_mesh_name(),
                self._config.get_read_data_name(),
                self._precice_vertex_ids,
                dt)

            read_data = {tuple(key): value for key, value in zip(self._owned_vertices.get_coordinates(), read_data)}
            read_data = communicate_shared_vertices(
                self._comm, self._fenics_vertices, self._send_map, self._recv_map, read_data)
        else:  # if there are no vertices, we return empty data
            read_data = None

        return copy.deepcopy(read_data)

    def write_data(self, write_function):
        """
        Writes data to preCICE. Depending on the dimensions of the simulation (2D-3D Coupling, 2D-2D coupling or
        Scalar/Vector write function) write_function is first converted into a format needed for preCICE.

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
        assert (self._write_function_type == determine_function_type(w_func, self._fenics_dims))
        assert (write_function.function_space() == self._write_function_space)

        if self._empty_rank:
            assert (self._is_parallel()
                    ), "having participants without coupling mesh nodes is only valid for parallel runs"

        write_function_type = determine_function_type(write_function, self._fenics_dims)
        assert (write_function_type in list(FunctionType))
        write_data = convert_fenics_to_precice(write_function, self._owned_vertices.get_local_ids())
        self._participant.write_data(
            self._config.get_coupling_mesh_name(),
            self._config.get_write_data_name(),
            self._precice_vertex_ids,
            write_data)

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
        if isinstance(write_object, Function):
            write_function_space = write_object.function_space()
            write_function = write_object
        elif isinstance(write_object, FunctionSpace):
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

        coords = function_space.tabulate_dof_coordinates()
        _, self._fenics_dims = coords.shape

        if self._coupling_type is CouplingMode.UNI_DIRECTIONAL_READ_COUPLING or \
                self._coupling_type is CouplingMode.BI_DIRECTIONAL_COUPLING:
            self._read_function_type = determine_function_type(read_function_space, self._fenics_dims)
            self._read_function_space = read_function_space

        if self._coupling_type is CouplingMode.UNI_DIRECTIONAL_WRITE_COUPLING or \
                self._coupling_type is CouplingMode.BI_DIRECTIONAL_COUPLING:
            # Ensure that function spaces of read and write functions are defined using the same mesh
            self._write_function_type = determine_function_type(write_function_space, self._fenics_dims)
            self._write_function_space = write_function_space

        # Ensure that function spaces of read and write functions use the same mesh
        if self._coupling_type is CouplingMode.BI_DIRECTIONAL_COUPLING:
            assert (self._read_function_space.mesh() is write_function_space.mesh()
                    ), "read_function_space and write_object need to be defined using the same mesh"

        if fixed_boundary:
            self._Dirichlet_Boundary = fixed_boundary

        if self._fenics_dims != self._participant.get_mesh_dimensions(self._config.get_coupling_mesh_name()):
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
        self._precice_vertex_ids = self._participant.set_mesh_vertices(
            self._config.get_coupling_mesh_name(), self._owned_vertices.get_coordinates())

        if (self._coupling_type is CouplingMode.UNI_DIRECTIONAL_READ_COUPLING or
                self._coupling_type is CouplingMode.BI_DIRECTIONAL_COUPLING) and self._is_parallel:
            # Determine shared vertices with neighbouring processes and get dictionaries for communication
            self._send_map, self._recv_map = get_communication_map(self._comm, self._read_function_space,
                                                                   self._owned_vertices,
                                                                   self._unowned_vertices)

        # Check for double boundary points
        if fixed_boundary:
            # create empty data for the sake of searching for duplicate points
            point_data = {tuple(key): None for key in self._owned_vertices.get_coordinates()}
            _ = filter_point_sources(point_data, fixed_boundary, warn_duplicate=True)

        # Set mesh connectivity information in preCICE to allow nearest-projection mapping
        if self._participant.requires_mesh_connectivity_for(self._config.get_coupling_mesh_name()):
            # Define a mapping between coupling vertices and their IDs in preCICE
            id_mapping = {
                key: value for key,
                value in zip(
                    self._owned_vertices.get_global_ids(),
                    self._precice_vertex_ids)}

            edge_vertex_ids, fenics_edge_ids = get_coupling_boundary_edges(
                function_space, coupling_subdomain, self._owned_vertices.get_global_ids(), id_mapping)

            # Surface coupling over 1D edges
            # TODO call set_mesh_edges, if using surface coupling. Otherwise does not make sense.
            self._participant.set_mesh_edges(self._config.get_coupling_mesh_name(), edge_vertex_ids)

            # Configure mesh connectivity (triangles from edges) for 2D simulations
            # TODO only enter code below, if using volume coupling. Otherwise does not make sense.
            if self._fenics_dims == 2:
                # Volume coupling over 2D triangles
                vertices = get_coupling_triangles(function_space, coupling_subdomain, fenics_edge_ids, id_mapping)
                self._participant.set_mesh_triangles(self._config.get_coupling_mesh_name(), vertices)
            else:
                print("Mesh connectivity information is not written for 3D cases.")

        if self._participant.requires_initial_data():
            if not write_function:
                raise Exception(
                    "preCICE requires you to write initial data. Please provide a write_function to initialize(...)")
            self.write_data(write_function)

        self._participant.initialize()

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
        Refer advance() in https://github.com/precice/python-bindings/blob/develop/cyprecice/cyprecice.pyx
        """
        self._first_advance_done = True
        self._participant.advance(dt)

    def finalize(self):
        """
        Finalizes the coupling via preCICE and the adapter. To be called at the end of the simulation.

        Notes
        -----
        Refer finalize() in https://github.com/precice/python-bindings/blob/develop/cyprecice/cyprecice.pyx
        """
        self._participant.finalize()

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
        Refer is_coupling_ongoing() in https://github.com/precice/python-bindings/blob/develop/cyprecice/cyprecice.pyx

        Returns
        -------
        tag : bool
            True if coupling is still going on and False if coupling has finished.
        """
        return self._participant.is_coupling_ongoing()

    def is_time_window_complete(self):
        """
        Tag to check if implicit iteration has converged.

        Notes
        -----
        Refer is_time_window_complete() in
        https://github.com/precice/python-bindings/blob/develop/cyprecice/cyprecice.pyx

        Returns
        -------
        tag : bool
            True if implicit coupling in the time window has converged and False if not converged yet.
        """
        return self._participant.is_time_window_complete()

    def get_max_time_step_size(self):
        """
        Get the maximum time step from preCICE.

        Notes
        -----
        Refer get_max_time_step_size() in
        https://github.com/precice/python-bindings/blob/develop/cyprecice/cyprecice.pyx

        Returns
        -------
        max_dt : double
            Maximum length of timestep to be computed by solver.
        """
        return self._participant.get_max_time_step_size()

    def requires_writing_checkpoint(self):
        """
        Tag to check if checkpoint needs to be written.

        Notes
        -----
        Refer requires_writing_checkpoint() in
        https://github.com/precice/python-bindings/blob/develop/cyprecice/cyprecice.pyx

        Returns
        -------
        tag : bool
            True if checkpoint needs to be written, False otherwise.
        """
        return self._participant.requires_writing_checkpoint()

    def requires_reading_checkpoint(self):
        """
        Tag to check if checkpoint needs to be read.

        Notes
        -----
        Refer requires_reading_checkpoint() in
        https://github.com/precice/python-bindings/blob/develop/cyprecice/cyprecice.pyx

        Returns
        -------
        tag : bool
            True if checkpoint needs to be written, False otherwise.
        """
        return self._participant.requires_reading_checkpoint()
