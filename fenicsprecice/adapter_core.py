"""
This module consists of helper functions used in the Adapter class. Names of the functions are self explanatory
"""

from fenics import SubDomain, Point, PointSource, vertices, FunctionSpace, Function, edges
import numpy as np
from enum import Enum
import logging
import hashlib
from mpi4py import MPI
import copy

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class VertexType(Enum):
    """
    Defines type of vertices that exist in the adapter.
    OWNED vertices are vertices on the coupling interface owned by this rank
    UNOWNED vertices are vertices on the coupling interface which are not owned by this rank. They are borrowed
        vertices from neigbouring ranks
    FENICS vertices are OWNED + UNOWNED vertices in the order as seen by FEniCS
    """
    OWNED = 153
    UNOWNED = 471
    FENICS = 557


class Vertices:
    """
    Vertices class provides a generic skeleton for vertices. A set of vertices has a set of global IDs, local IDs and
    coordinates as defined in FEniCS.
    """

    def __init__(self, vertex_type):
        self._vertex_type = vertex_type
        self._global_ids = None
        self._local_ids = None
        self._coordinates = None

    def set_global_ids(self, ids):
        self._global_ids = ids

    def set_local_ids(self, ids):
        self._local_ids = ids

    def set_coordinates(self, coords):
        self._coordinates = coords

    def get_global_ids(self):
        return copy.deepcopy(self._global_ids)

    def get_local_ids(self):
        return copy.deepcopy(self._local_ids)

    def get_coordinates(self):
        return copy.deepcopy(self._coordinates)


class FunctionType(Enum):
    """
    Defines scalar- and vector-valued function.
    Used in assertions to check if a FEniCS function is scalar or vector.
    """
    SCALAR = 0  # scalar valued function
    VECTOR = 1  # vector valued function


class CouplingMode(Enum):
    """
    Defines the type of coupling being used.
    Options are: Bi-directional coupling, Uni-directional Write Coupling, Uni-directional Read Coupling
    Used in assertions to check which type of coupling is done
    """
    BI_DIRECTIONAL_COUPLING = 4
    UNI_DIRECTIONAL_WRITE_COUPLING = 5
    UNI_DIRECTIONAL_READ_COUPLING = 6


def determine_function_type(input_obj):
    """
    Determines if the function is scalar- or vector-valued based on rank evaluation.

    Parameters
    ----------
    input_obj :
        A FEniCS function.

    Returns
    -------
    tag : bool
        0 if input_function is SCALAR and 1 if input_function is VECTOR.
    """
    if isinstance(input_obj, FunctionSpace):  # scalar-valued functions have rank 0 is FEniCS
        if input_obj.num_sub_spaces() == 0:
            return FunctionType.SCALAR
        elif input_obj.num_sub_spaces() == 2:
            return FunctionType.VECTOR
    elif isinstance(input_obj, Function):
        if input_obj.value_rank() == 0:
            return FunctionType.SCALAR
        elif input_obj.value_rank() == 1:
            return FunctionType.VECTOR
        else:
            raise Exception("Error determining type of given dolfin Function")
    else:
        raise Exception("Error determining type of given dolfin FunctionSpace")


def filter_point_sources(point_sources, filter_out, warn_duplicate=True):
    """
    Filter dictionary of PointSources (point_sources) with respect to a given domain (filter_out). If a PointSource
    is applied at a point inside of the given domain (filter_out), this PointSource will be removed from dictionary.

    Parameters
    ----------
    point_sources : python dictionary
        Dictionary containing coordinates and associated PointSources {(point_x, point_y): PointSource, ...}.
    filter_out: FEniCS domain
        Defines the domain where PointSources should be filtered out.
    warn_duplicate: bool
        Set False to surpress warnings, if double-boundary points are filtered out.

    Returns
    -------
    filtered_point_sources : python dictionary
        A dictionary with the filtered PointSources.
    """
    filtered_point_sources = dict()

    for point in point_sources.keys():
        # Filter double boundary points to avoid instabilities and create PointSource
        if not filter_out.inside(point, 1):
            filtered_point_sources[point] = point_sources[point]
        elif warn_duplicate:
            logger.warning("Found a double-boundary point at {location}.".format(location=point))

    return filtered_point_sources


def convert_fenics_to_precice(fenics_function, local_ids):
    """
    Converts data of type dolfin.Function into Numpy array for all x and y coordinates on the boundary.

    Parameters
    ----------
    fenics_function : FEniCS function
        A FEniCS function referring to a physical variable in the problem.
    local_ids: numpy array
        Array of local indices of vertices on the coupling interface and owned by this rank.

    Returns
    -------
    precice_data : array_like
        Array of FEniCS function values at each point on the boundary.
    """

    if not isinstance(fenics_function, Function):
        raise Exception("Cannot handle data type {}".format(type(fenics_function)))

    precice_data = []

    if fenics_function.function_space().num_sub_spaces() > 0:
        dims = fenics_function.function_space().num_sub_spaces()
        sampled_data = fenics_function.compute_vertex_values().reshape([dims, -1])
    else:
        sampled_data = fenics_function.compute_vertex_values()

    if len(local_ids):
        if fenics_function.function_space().num_sub_spaces() > 0:  # function space is VectorFunctionSpace
            for lid in local_ids:
                precice_data.append(sampled_data[:, lid])
        else:  # function space is FunctionSpace (scalar)
            for lid in local_ids:
                precice_data.append(sampled_data[lid])
    else:
        precice_data = np.array([])

    return np.array(precice_data)


def get_fenics_vertices(function_space, coupling_subdomain, dims):
    """
    Extracts vertices which FEniCS accesses on this rank and which lie on the given coupling domain, from a given
    function space.

    Parameters
    ----------
    function_space : FEniCS function space
        Function space on which the finite element problem definition lives.
    coupling_subdomain : FEniCS Domain
        Subdomain consists of only the coupling interface region.
    dims : int
        Dimension of problem.

    Returns
    -------
    lids : numpy array
        Array of local ids of fenics vertices.
    gids : numpy array
        Array of global ids of fenics vertices.
    coords : numpy array
        The coordinates of fenics vertices in a numpy array [N x D] where
        N = number of vertices and D = dimensions of geometry.
    """

    if not issubclass(type(coupling_subdomain), SubDomain):
        raise Exception("No correct coupling interface defined! Given coupling domain is not of type dolfin Subdomain")

    # Get mesh from FEniCS function space
    mesh = function_space.mesh()

    # Get coordinates and global IDs of all vertices of the mesh  which lie on the coupling boundary.
    # These vertices include shared (owned + unowned) and non-shared vertices in a parallel setting
    lids, gids, coords = [], [], []
    for v in vertices(mesh):
        if coupling_subdomain.inside(v.point(), True):
            lids.append(v.index())
            gids.append(v.global_index())
            if dims == 2:
                coords.append([v.x(0), v.x(1)])
            if dims == 3:
                coords.append([v.x(0), v.x(1), v.x(2)])

    return np.array(lids), np.array(gids), np.array(coords)


def get_owned_vertices(function_space, coupling_subdomain, dims):
    """
    Extracts vertices which this rank owns and which lie on the given coupling domain, from a given function space .
    Parameters
    ----------
    function_space : FEniCS function space
        Function space on which the finite element problem definition lives.
    coupling_subdomain : FEniCS Domain
        Subdomain consists of only the coupling interface region.
    dims : int
        Dimension of problem.

    Returns
    -------
    lids : numpy array
        Array of local ids of owned vertices.
    gids : numpy array
        Array of global ids of owned vertices.
    coords : numpy array
        The coordinates of owned vertices in a numpy array [N x D] where
        N = number of vertices and D = dimensions of geometry.
    """

    if not issubclass(type(coupling_subdomain), SubDomain):
        raise Exception("No correct coupling interface defined! Given coupling domain is not of type dolfin Subdomain")

    # DoF coordinates of owned vertices (same as physical vertices for this particular FEniCS function call)
    all_dofs = function_space.tabulate_dof_coordinates()

    phy_dofs = all_dofs
    # For a VectorFunctionSpace each DoF occurs as many times as the components of quantities
    if function_space.num_sub_spaces() == dims:
        phy_dofs = all_dofs[::dims]

    dofs = []
    # Filter DoFs which are on the coupling interface
    for dof in phy_dofs:
        if coupling_subdomain.inside(dof, True):
            dofs.append(dof)

    dofs = np.array(dofs)

    # Get mesh from FEniCS function space
    mesh = function_space.mesh()

    # Filter vertices which lie on coupling interface
    coupling_vertices = []
    for v in vertices(mesh):
        if coupling_subdomain.inside(v.point(), True):
            coupling_vertices.append(v)

    # Get coordinates and global IDs of all vertices of the mesh  which lie on the coupling boundary.
    # These vertices include shared (owned + unowned) and non-shared vertices in a parallel setting
    gids, lids, coords = [], [], []
    coord = None
    for v in coupling_vertices:
        if dims == 2:
            coord = [v.x(0), v.x(1)]
        elif dims == 3:
            coord = [v.x(0), v.x(1), v.x(2)]

        for dof in dofs:
            if (dof == coord).all():
                gids.append(v.global_index())
                lids.append(v.index())
                coords.append(coord)

    return np.array(lids), np.array(gids), np.array(coords)


def get_unowned_vertices(function_space, coupling_subdomain, dims):
    """
    Extracts vertices which this rank does not own but shares and which lie on a given coupling domain, from a given
    function space.
    Parameters
    ----------
    function_space : FEniCS function space
        Function space on which the finite element problem definition lives.
    coupling_subdomain : FEniCS Domain
        Subdomain consists of only the coupling interface region.
    dims : int
        Dimension of problem.

    Returns
    -------
    gids : numpy array
        Array of global ids of unowned vertices.
    """

    if not issubclass(type(coupling_subdomain), SubDomain):
        raise Exception("No correct coupling interface defined! Given coupling domain is not of type dolfin Subdomain")

    # DoF coordinates of owned vertices (same as physical vertices for this particular FEniCS function call)
    all_dofs = function_space.tabulate_dof_coordinates()

    phy_dofs = all_dofs
    # For a VectorFunctionSpace each DoF occurs as many times as the components of quantities
    if function_space.num_sub_spaces() == dims:
        phy_dofs = all_dofs[::dims]

    dofs = []
    # Filter DoFs which are on the coupling interface
    for dof in phy_dofs:
        if coupling_subdomain.inside(dof, True):
            dofs.append(dof)

    # Get mesh from FEniCS function space
    mesh = function_space.mesh()

    # Filter vertices which lie on coupling interface
    coupling_verts = []
    for v in vertices(mesh):
        if coupling_subdomain.inside(v.point(), True):
            coupling_verts.append(v)

    # Get coordinates and global IDs of all vertices of the mesh  which lie on the coupling boundary.
    # These vertices include shared (owned + unowned) and non-shared vertices in a parallel setting
    gids = []
    coord = None
    for v in coupling_verts:
        ownership = False
        if dims == 2:
            coord = [v.x(0), v.x(1)]
        elif dims == 3:
            coord = [v.x(0), v.x(1), v.x(2)]

        for dof in dofs:
            if (dof == coord).all():
                ownership = True
                break

        if ownership is False:
            gids.append(v.global_index())

    return np.array(gids)


def get_coupling_boundary_edges(function_space, coupling_subdomain, global_ids, id_mapping):
    """
    Extracts edges of mesh which lie on the coupling boundary.

    Parameters
    ----------
    function_space : FEniCS function space
        Function space on which the finite element problem definition lives.
    coupling_subdomain : FEniCS Domain
        FEniCS domain of the coupling interface region.
    global_ids: numpy_array
        Array of global IDs of vertices owned by this rank.
    id_mapping : python dictionary
        Dictionary mapping preCICE vertex IDs to FEniCS global vertex IDs.

    Returns
    -------
    vertices1_ids : numpy array
        Array of first vertex of each edge.
    vertices2_ids : numpy array
        Array of second vertex of each edge.
    """

    def edge_is_on(subdomain, edge):
        """
        Check whether edge lies within subdomain
        """
        assert(len(list(vertices(edge))) == 2)
        return all([subdomain.inside(v.point(), True) for v in vertices(edge)])

    vertices1_ids = []
    vertices2_ids = []

    for edge in edges(function_space.mesh()):
        if edge_is_on(coupling_subdomain, edge):
            v1, v2 = list(vertices(edge))
            if v1.global_index() in global_ids and v2.global_index() in global_ids:
                vertices1_ids.append(id_mapping[v1.global_index()])
                vertices2_ids.append(id_mapping[v2.global_index()])

    vertices1_ids = np.array(vertices1_ids)
    vertices2_ids = np.array(vertices2_ids)

    return vertices1_ids, vertices2_ids


def get_forces_as_point_sources(fixed_boundary, function_space, data):
    """
    Creating two dicts of PointSources that can be applied to the assembled system. Applying filter_point_source to
    avoid forces being applied to already existing Dirichlet BC, since this would lead to an overdetermined system
    that cannot be solved.

    Parameters
    ----------
    fixed_boundary : FEniCS domain
        FEniCS domain consisting of a fixed boundary condition. For example in FSI cases usually the solid body is fixed
        at one end.
    function_space : FEniCS function space
        Function space on which the finite element problem definition lives.
    data : a numpy array of PointSource
        FEniCS PointSource data carrying forces

    Returns
    -------
    x_forces : list
        Dictionary carrying X component of forces with reference to each point on the coupling interface.
    y_forces : list
        Dictionary carrying Y component of forces with reference to each point on the coupling interface.
    """
    x_forces = dict()  # dict of PointSources for Forces in x direction
    y_forces = dict()  # dict of PointSources for Forces in y direction

    fenics_vertices = np.array(list(data.keys()))
    nodal_data = np.array(list(data.values()))

    # Check for shape of coupling_mesh_vertices and raise Assertion for 3D
    n_vertices, _ = fenics_vertices.shape

    vertices_x = fenics_vertices[:, 0]
    vertices_y = fenics_vertices[:, 1]

    for i in range(n_vertices):
        px, py = vertices_x[i], vertices_y[i]
        key = (px, py)
        x_forces[key] = PointSource(function_space.sub(0), Point(px, py), nodal_data[i, 0])
        y_forces[key] = PointSource(function_space.sub(1), Point(px, py), nodal_data[i, 1])

    # Avoid application of PointSource and Dirichlet boundary condition at the same point by filtering
    x_forces = filter_point_sources(x_forces, fixed_boundary, warn_duplicate=False)
    y_forces = filter_point_sources(y_forces, fixed_boundary, warn_duplicate=False)

    return x_forces.values(), y_forces.values()  # don't return dictionary, but list of PointSources


def get_communication_map(comm, function_space, owned_vertices, unowned_vertices):
    """
    Determine which vertices along the coupling boundary are shared with neighbouring processes. This function creates
    a map of vertices to be sent and received from neighbouring processes. This map is used for non-blocking
    communication.

    Parameters
    ----------
    comm : Object of class MPI.COMM_WORLD from mpi4py package
        A predefined intra-communicator instance available in mpi4py.
    function_space : FEniCS function space
        Function space on which the finite element problem definition lives.
    owned_vertices : Object of class Vertices
        Vertices owned by this rank.
    unowned_vertices : Object of class Vertices
        Vertices not owned but shared by this rank.

    Returns
    -------
    send_data : dict_like
        A dictionary with global IDs of vertices whose data needs to be sent as the keys, and the ranks to which the
        data needs to be sent as the values.
    recv_data : dict_like
        A dictionary with global IDs of vertices on which data needs to be received as keys, and the ranks from which
        the data needs to be received as values.

    """
    owned_gids = owned_vertices.get_global_ids()
    unowned_gids = unowned_vertices.get_global_ids()

    # Get rank
    rank = comm.Get_rank()

    # Get ranks which are neighbours of this rank from the DoFMap in FEniCS
    neigh_ranks = function_space.dofmap().neighbours()

    # Receive ownership data of ranks which are neighbors of this rank
    recv_reqs = []
    for neigh in neigh_ranks:
        recv_hashtag = hashlib.sha256()
        recv_hashtag.update((str(neigh) + str(rank)).encode('utf-8'))
        recv_tag = int(recv_hashtag.hexdigest()[:6], base=16)
        recv_reqs.append(comm.irecv(source=neigh, tag=recv_tag))

    # Send ownership data to ranks which are neighbors of this rank
    send_reqs = []
    for neigh in neigh_ranks:
        send_hashtag = hashlib.sha256()
        send_hashtag.update((str(rank) + str(neigh)).encode('utf-8'))
        send_tag = int(send_hashtag.hexdigest()[:6], base=16)
        req = comm.isend(owned_gids, dest=neigh, tag=send_tag)
        send_reqs.append(req)

    # Wait for all non-blocking communication operations to complete
    MPI.Request.Waitall(send_reqs)

    ownership_data = dict()
    # Collect received ownership data of other ranks
    counter = 0
    for neigh in neigh_ranks:
        ownership_data[neigh] = recv_reqs[counter].wait()
        counter += 1

    # Create dictionary: keys are Global IDs of vertices not owned by this rank, values are ranks from which
    # data on these vertices needs to be received
    recv_data = dict()
    for neigh in neigh_ranks:
        for o_gid in ownership_data[neigh]:
            for uo_gid in unowned_gids:
                if o_gid == uo_gid:
                    recv_data[uo_gid] = neigh

    # Receive ownership data of ranks which are neighbors of this rank
    recv_reqs = []
    for neigh in neigh_ranks:
        recv_hashtag = hashlib.sha256()
        recv_hashtag.update((str(neigh) + str(rank)).encode('utf-8'))
        recv_tag = int(recv_hashtag.hexdigest()[:6], base=16)
        recv_reqs.append(comm.irecv(source=neigh, tag=recv_tag))

    # Send non-ownership data to ranks which are neighbors of this rank
    send_reqs = []
    for neigh in neigh_ranks:
        send_hashtag = hashlib.sha256()
        send_hashtag.update((str(rank) + str(neigh)).encode('utf-8'))
        send_tag = int(send_hashtag.hexdigest()[:6], base=16)
        req = comm.isend(unowned_gids, dest=neigh, tag=send_tag)
        send_reqs.append(req)

    # Wait for all non-blocking communication operations to complete
    MPI.Request.Waitall(send_reqs)

    nonownership_data = dict()
    # Collect received ownership data of other ranks
    counter = 0
    for neigh in neigh_ranks:
        nonownership_data[neigh] = recv_reqs[counter].wait()
        counter += 1

    # Create dictionary: keys are Global IDs of vertices not owned by this rank, values are ranks from which
    # data on these vertices needs to be received
    send_data = dict()
    for neigh in neigh_ranks:
        for uo_gid in nonownership_data[neigh]:
            for o_gid in owned_gids:
                if uo_gid == o_gid:
                    send_data[o_gid] = neigh

    return send_data, recv_data


def communicate_shared_vertices(comm, fenics_vertices, send_pts, recv_pts, coupling_data):
    """
    Triggers asynchronous communication between ranks of this solver to exchange data for shared vertices. Rank owning a
    shared vertex sends latest data to all ranks it is sharing this vertex with.

    Parameters
    ----------
    comm : Object of class MPI.COMM_WORLD from mpi4py package
        A predefined intra-communicator instance available in mpi4py.
    fenics_vertices : Object of class Vertices
        Vertices owned and shared by this rank, as seen by FEniCS
    send_pts : dict_like
        A dictionary with global IDs of vertices whose data needs to be sent as the keys, and the ranks to which the
        data needs to be sent as the values.
    recv_pts : dict_like
        A dictionary with global IDs of vertices on which data needs to be received as keys, and the ranks from which
        the data needs to be received as values.
    coupling_data : dict_like
        The coupling data. A dictionary containing nodal data with vertex coordinates as key and associated data as
            values.

    Returns
    -------
    fenics_data : array_like
        The updated coupling data. A dictionary containing nodal data with vertex coordinates of all seen vertices as
        the keys, and the associated data as values.

    """
    fenics_coords = fenics_vertices.get_coordinates()
    fenics_gids = fenics_vertices.get_global_ids()

    # Get rank
    rank = comm.Get_rank()

    # Attach data read from preCICE to appropriate ids in FEniCS style array (which includes duplicates)
    for coord in fenics_coords:
        if tuple(coord) not in coupling_data.keys():
            coupling_data[tuple(coord)] = None

    recv_reqs = []
    if recv_pts:
        for recv_gid, src in recv_pts.items():
            hash_tag = hashlib.sha256()
            hash_tag.update((str(src) + str(recv_gid) + str(rank)).encode('utf-8'))
            tag = int(hash_tag.hexdigest()[:6], base=16)
            recv_reqs.append(comm.irecv(source=src, tag=tag))

    send_reqs = []
    if send_pts:
        for send_gid, dests in send_pts.items():
            # Make sure that there is only one send_lid
            send_lid = int(np.where(fenics_gids == send_gid)[0][0])
            for dest in [dests]:
                hash_tag = hashlib.sha256()
                hash_tag.update((str(rank) + str(send_gid) + str(dest)).encode('utf-8'))
                tag = int(hash_tag.hexdigest()[:6], base=16)
                req = comm.isend(coupling_data[tuple(fenics_coords[send_lid])], dest=dest, tag=tag)
                send_reqs.append(req)

    # Wait for all non-blocking communications to complete
    MPI.Request.Waitall(send_reqs)

    # Attach received data into the existing FEniCS style data array
    counter = 0
    for recv_gid in recv_pts.keys():
        recv_lid = int(np.where(fenics_gids == recv_gid)[0][0])
        coupling_data[tuple(fenics_coords[recv_lid])] = recv_reqs[counter].wait()
        counter += 1

    return coupling_data
