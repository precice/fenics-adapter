"""
This module consists of helper functions used in the Adapter class. Names of the functions are self explanatory
"""

import dolfin
from dolfin import SubDomain, Point, PointSource, vertices, vertex_to_dof_map, dof_to_vertex_map
from fenics import FunctionSpace, VectorFunctionSpace, Function, interpolate
import numpy as np
from enum import Enum
import logging
import hashlib
from mpi4py import MPI

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class FunctionType(Enum):
    """
    Defines scalar- and vector-valued function.
    Used in assertions to check if a FEniCS function is scalar or vector.
    """
    SCALAR = 0  # scalar valued function
    VECTOR = 1  # vector valued function


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
    if type(input_obj) == FunctionSpace:  # scalar-valued functions have rank 0 is FEniCS
        if input_obj.num_sub_spaces() == 0:
            return FunctionType.SCALAR
        elif input_obj.num_sub_spaces() == 2:
            return FunctionType.VECTOR
    elif type(input_obj) == Function:
        if input_obj.value_rank() == 0:
            return FunctionType.SCALAR
        elif input_obj.value_rank() == 1:
            return FunctionType.VECTOR
        else:
            raise Exception("Error determining type of given dolfin Function")
    else:
        raise Exception("Error determining type of given dolfin FunctionSpace")


def filter_point_sources(point_sources, filter_out):
    """
    Filter dictionary of PointSources (point_sources) with respect to a given domain (filter_out). If a PointSource
    is applied at a point inside of the given domain (filter_out), this PointSource will be removed from dictionary.

    Parameters
    ----------
    point_sources : python dictionary
        Dictionary containing coordinates and associated PointSources {(point_x, point_y): PointSource, ...}.
    filter_out: FEniCS domain
        Defines the domain where PointSources should be filtered out.

    Returns
    -------
    filtered_point_sources : python dictionary
        A dictionary with the filtered PointSources.
    """
    filtered_point_sources = dict()

    for point in point_sources.keys():
        # Filter double boundary points to avoid instabilities and create PointSource
        if filter_out.inside(point, 1):
            print("Found a double-boundary point at {location}.".format(location=point))
        else:
            filtered_point_sources[point] = point_sources[point]

    return filtered_point_sources


def convert_fenics_to_precice(function, coupling_subdomain, dimensions):
    """
    Converts data of type dolfin.Function into Numpy array for all x and y coordinates on the boundary.

    Parameters
    ----------
    function : FEniCS function
        A FEniCS function referring to a physical variable in the problem.

    Returns
    -------
    array : array_like
        Array of FEniCS function values at each point on the boundary.
    """

    if type(function) is not dolfin.Function:
        raise Exception("Cannot handle data type {}".format(type(function)))

    if function.function_space().num_sub_spaces() > 0:  # function space is a VectorFunctionSpace
        linear_space = VectorFunctionSpace(function.function_space().mesh(), "P", 1)
    else:  # function space is scalar
        linear_space = FunctionSpace(function.function_space().mesh(), "P", 1)

    _, lids, _ = get_owned_coupling_boundary_vertices(linear_space, coupling_subdomain)

    vertex_vals = function.compute_vertex_values(function.function_space().mesh())

    result_vals = []
    if len(lids):
        if function.function_space().num_sub_spaces() > 0:  # function space is VectorFunctionSpace
            for lid in lids:
                result_vals.append([vertex_vals[lid], vertex_vals[lid+1]])
        else:  # function space is FunctionSpace (scalar)
            for lid in lids:
                result_vals.append(vertex_vals[lid])
    else:
        result_vals = np.array([])

    return np.array(result_vals)


def get_fenics_coupling_boundary_vertices(function_space, coupling_subdomain):
    """
    Extracts vertices which this rank owns from a given function space which lie on the given coupling domain.

    Parameters
    ----------
    function_space : FEniCS function space
        Function space on which the finite element problem definition lives.
    coupling_subdomain : FEniCS Domain
        Subdomain consists of only the coupling interface region.

    Returns
    -------
    fenics_vertices : numpy array
        Array consisting of all vertices lying on the coupling interface.
    coordinates : array_like
        The coordinates of the vertices in a numpy array [N x D] where
        N = number of vertices and D = dimensions of geometry.
    n : int
        Number of vertices on the coupling interface.
    """

    if not issubclass(type(coupling_subdomain), SubDomain):
        raise Exception("No correct coupling interface defined! Given coupling domain is not of type dolfin Subdomain")

    # Get mesh from FEniCS function space
    mesh = function_space.mesh()

    # Get coordinates and global IDs of all vertices of the mesh  which lie on the coupling boundary.
    # These vertices include shared (owned + unowned) and non-shared vertices in a parallel setting
    fenics_gids, fenics_lids, fenics_coords = [], [], []
    for v in vertices(mesh):
        if coupling_subdomain.inside(v.point(), True):
            fenics_gids.append(v.global_index())
            fenics_lids.append(v.index())
            fenics_coords.append([v.x(0), v.x(1)])

    return np.array(fenics_gids), np.array(fenics_lids), np.array(fenics_coords)


def get_owned_coupling_boundary_vertices(function_space, coupling_subdomain):
    """
        Extracts vertices which this rank owns from a given function space which lie on the given coupling domain.

        Parameters
        ----------
        function_space : FEniCS function space
            Function space on which the finite element problem definition lives.
        coupling_subdomain : FEniCS Domain
            Subdomain consists of only the coupling interface region.

        Returns
        -------
        fenics_vertices : numpy array
            Array consisting of all vertices lying on the coupling interface.
        coordinates : array_like
            The coordinates of the vertices in a numpy array [N x D] where
            N = number of vertices and D = dimensions of geometry.
        n : int
            Number of vertices on the coupling interface.
        """

    if not issubclass(type(coupling_subdomain), SubDomain):
        raise Exception("No correct coupling interface defined! Given coupling domain is not of type dolfin Subdomain")

    # DoF coordinates of owned vertices (same as physical vertices for this particular FEniCS function call)
    dofs = function_space.tabulate_dof_coordinates()

    # Get mesh from FEniCS function space
    mesh = function_space.mesh()

    # Get coordinates and global IDs of all vertices of the mesh  which lie on the coupling boundary.
    # These vertices include shared (owned + unowned) and non-shared vertices in a parallel setting
    owned_gids, owned_lids, owned_coords = [], [], []
    for v in vertices(mesh):
        if coupling_subdomain.inside(v.point(), True):
            dof_nm1 = None  # If function_space is VectorFunctionSpace then each vertex has multiple DoFs
            for dof in dofs:
                if (dof == [v.x(0), v.x(1)]).all() and (dof != dof_nm1).any():
                    owned_gids.append(v.global_index())
                    owned_lids.append(v.index())
                    owned_coords.append([v.x(0), v.x(1)])
                dof_nm1 = dof

    return np.array(owned_gids), np.array(owned_lids), np.array(owned_coords)


def get_unowned_coupling_boundary_vertices(function_space, coupling_subdomain):
    """
        Extracts vertices which this rank owns from a given function space which lie on the given coupling domain.

        Parameters
        ----------
        function_space : FEniCS function space
            Function space on which the finite element problem definition lives.
        coupling_subdomain : FEniCS Domain
            Subdomain consists of only the coupling interface region.

        Returns
        -------
        fenics_vertices : numpy array
            Array consisting of all vertices lying on the coupling interface.
        coordinates : array_like
            The coordinates of the vertices in a numpy array [N x D] where
            N = number of vertices and D = dimensions of geometry.
        n : int
            Number of vertices on the coupling interface.
        """

    if not issubclass(type(coupling_subdomain), SubDomain):
        raise Exception("No correct coupling interface defined! Given coupling domain is not of type dolfin Subdomain")

    # DoF coordinates of owned vertices (same as physical vertices for this particular FEniCS function call)
    dofs = function_space.tabulate_dof_coordinates()

    # Get mesh from FEniCS function space
    mesh = function_space.mesh()

    # Get coordinates and global IDs of all vertices of the mesh  which lie on the coupling boundary.
    # These vertices include shared (owned + unowned) and non-shared vertices in a parallel setting
    unowned_gids, unowned_lids = [], []
    ownership = False
    for v in vertices(mesh):
        if coupling_subdomain.inside(v.point(), True):
            dof_nm1 = None  # If function_space is VectorFunctionSpace then each vertex has multiple DoFs
            for dof in dofs:
                if (dof == [v.x(0), v.x(1)]).all() and (dof != dof_nm1).any():
                    ownership = True
                    break
                dof_nm1 = dof

            if ownership is False:
                unowned_gids.append(v.global_index())
                unowned_lids.append(v.index())

    return np.array(unowned_gids), np.array(unowned_lids)


def are_connected_by_edge(v1, v2):
    """
    Checks if vertices are connected by an edge.

    Parameters
    ----------
    v1 : dolfin.vertex
        Vertex 1 of the edge
    v2 : dolfin.vertex
        Vertex 2 of the edge

    Returns
    -------
    tag : bool
        True is v1 and v2 are connected by edge and False if not connected
    """
    for edge1 in dolfin.edges(v1):
        for edge2 in dolfin.edges(v2):
            if edge1.index() == edge2.index():  # Vertices are connected by edge
                return True
    return False


def get_coupling_boundary_edges(function_space, coupling_subdomain, id_mapping):
    """
    Extracts edges of mesh which lie on the coupling boundary.

    Parameters
    ----------
    function_space : FEniCS function space
        Function space on which the finite element problem definition lives.
    coupling_subdomain : FEniCS Domain
        FEniCS domain of the coupling interface region.
    id_mapping : python dictionary
        Dictionary mapping preCICE vertex IDs to FEniCS global vertex IDs.

    Returns
    -------
    vertices1_ids : numpy array
        Array of first vertex of each edge.
    vertices2_ids : numpy array
        Array of second vertex of each edge.
    """
    # preCICE only sees non-duplicate global vertices
    dofs = function_space.tabulate_dof_coordinates()
    local_to_global_map = function_space.dofmap().tabulate_local_to_global_dofs()
    local_to_global_unowned = function_space.dofmap().local_to_global_unowned()
    global_ids = [i for i in local_to_global_map if i not in local_to_global_unowned]

    points = dict()

    for v1 in dofs:
        if coupling_subdomain.inside(v1, True):
            points[v1] = []

    for v1 in points.keys():
        for v2 in points.keys():
            if are_connected_by_edge(v1, v2):
                points[v1] = v2
                points[v2] = v1

    vertices1_ids = []
    vertices2_ids = []

    for v1, v2 in points.items():
        if v1 is not v2:
            vertices1_ids.append(id_mapping[v1.global_index()])
            vertices2_ids.append(id_mapping[v2.global_index()])

    vertices1_ids = np.array(vertices1_ids)
    vertices2_ids = np.array(vertices2_ids)

    return vertices1_ids, vertices2_ids


def get_forces_as_point_sources(fixed_boundary, function_space, coupling_mesh_vertices, data):
    """
    Creating two dicts of PointSources that can be applied to the assembled system. Appling filter_point_source to
    avoid forces being applied to already existing Dirichlet BC, since this would lead to an overdetermined system
    that cannot be solved.

    Parameters
    ----------
    fixed_boundary : FEniCS domain
        FEniCS domain consisting of a fixed boundary condition. For example in FSI cases usually the solid body is fixed
        at one end.
    function_space : FEniCS function space
        Function space on which the finite element problem definition lives.
    coupling_mesh_vertices : numpy.ndarray
        The coordinates of the vertices on the coupling interface. Coordinates of vertices are stored in a
        numpy array [N x D] where N = number of vertices and D = dimensions of geometry
    data : a numpy array of PointSource
        FEniCS PointSource data carrying forces

    Returns
    -------
    x_forces : list
        Dictionary carrying X component of forces with reference to each point on the coupling interface.
    y_forces : list
        Dictionary carrying Y component of forces with reference to each point on the coupling interface.
    """
    # PointSources are scalar valued, therefore we need an individual scalar valued PointSource for each dimension in a vector-valued setting
    # TODO: a vector valued PointSource would be more straightforward, but does not exist (as far as I know)

    x_forces = dict()  # dict of PointSources for Forces in x direction
    y_forces = dict()  # dict of PointSources for Forces in y direction

    # Check for shape of coupling_mesh_vertices and raise Assertion for 3D
    n_vertices, dims = coupling_mesh_vertices.shape

    vertices_x = coupling_mesh_vertices[:, 0]
    vertices_y = coupling_mesh_vertices[:, 1]

    for i in range(n_vertices):
        px, py = vertices_x[i], vertices_y[i]
        key = (px, py)
        x_forces[key] = PointSource(function_space.sub(0), Point(px, py), data[i, 0])
        y_forces[key] = PointSource(function_space.sub(1), Point(px, py), data[i, 1])

    # Avoid application of PointSource and Dirichlet boundary condition at the same point by filtering
    x_forces = filter_point_sources(x_forces, fixed_boundary)
    y_forces = filter_point_sources(y_forces, fixed_boundary)

    return x_forces.values(), y_forces.values()  # don't return dictionary, but list of PointSources


def determine_shared_vertices(comm, rank, function_space, owned_gids, unowned_gids):
    """
    Determine which vertices along the coupling boundary are shared with neighbouring processes
    Parameters
    ----------
    fenics_lids
    comm
    rank
    function_space :

    Returns
    -------

    """
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


def communicate_shared_vertices(comm, rank, fenics_gids, owned_coords, fenics_coords, owned_data,
                                send_pts, recv_pts):
    """

    Parameters
    ----------
    comm
    rank
    dofmap
    precice_data
    to_send_ids
    to_recv_ids

    Returns
    -------

    """
    # Create fenics type shared data array for communication
    fenics_data = dict.fromkeys(tuple(key) for key in fenics_coords)

    # Attach data read from preCICE to appropriate ids in FEniCS style array (which includes duplicates)
    for coord in fenics_coords:
        for owned_coord in owned_coords:
            if (coord == owned_coord).all():
                fenics_data[tuple(coord)] = owned_data[tuple(owned_coord)]

    recv_reqs = []
    if recv_pts:
        for recv_gid, src in recv_pts.items():
            hash_tag = hashlib.sha256()
            hash_tag.update((str(src) + str(recv_gid) + str(rank)).encode('utf-8'))
            tag = int(hash_tag.hexdigest()[:6], base=16)
            recv_reqs.append(comm.irecv(source=src, tag=tag))
    else:
        print("Rank {}: Nothing to Receive".format(rank))

    send_reqs = []
    if send_pts:
        for send_gid, dests in send_pts.items():
            send_lid = int(np.where(fenics_gids == send_gid)[0][0])
            for dest in [dests]:
                hash_tag = hashlib.sha256()
                hash_tag.update((str(rank) + str(send_gid) + str(dest)).encode('utf-8'))
                tag = int(hash_tag.hexdigest()[:6], base=16)
                req = comm.isend(fenics_data[tuple(fenics_coords[send_lid])], dest=dest, tag=tag)
                send_reqs.append(req)
    else:
        print("Rank {}: Nothing to Send".format(rank))

    # Wait for all non-blocking communications to complete
    MPI.Request.Waitall(send_reqs)

    # Attach received data into the existing FEniCS style data array
    counter = 0
    for recv_gid in recv_pts.keys():
        recv_lid = int(np.where(fenics_gids == recv_gid)[0][0])
        fenics_data[tuple(fenics_coords[recv_lid])] = recv_reqs[counter].wait()
        counter += 1

    return fenics_data
