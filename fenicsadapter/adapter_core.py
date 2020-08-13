"""
This module consists of helper functions used in the Adapter class. Names of the functions are self explanatory
"""

import dolfin
from dolfin import SubDomain, Point, PointSource
from fenics import FunctionSpace, VectorFunctionSpace, Function
import numpy as np
from enum import Enum
import logging

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


def convert_fenics_to_precice(data, sample_points):
    """
    Converts data of type dolfin.Function into Numpy array for all x and y coordinates on the boundary.

    Parameters
    ----------
    data : FEniCS function
        A FEniCS function referring to a physical variable in the problem.
    sample_points : array_like
        The coordinates of the vertices in a numpy array [N x D] where
        N = number of vertices and D = dimensions of geometry.

    Returns
    -------
    array : array_like
        Array of FEniCS function values at each point on the boundary.
    """
    if type(data) is dolfin.Function:
        x_all, y_all = sample_points[:, 0], sample_points[:, 1]
        return np.array([data(x, y) for x, y in zip(x_all, y_all)])
    else:
        raise Exception("Cannot handle data type %s" % type(data))


def get_coupling_boundary_vertices(mesh_fenics, coupling_subdomain, fenics_dimensions, dimensions):
    """
    Extracts vertices from a given mesh which lie on the  given coupling domain.

    Parameters
    ----------
    mesh_fenics : FEniCS Mesh
        Mesh of complete domain.
    coupling_subdomain : FeniCS Domain
        Subdomain consists of only the coupling interface region.
    fenics_dimensions : int
        Dimensions of FEniCS case setup.
    dimensions : int
        Dimensions of coupling case setup.

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
    n = 0
    fenics_vertices = []
    vertices_x = []
    vertices_y = []
    vertices_z = []

    if not issubclass(type(coupling_subdomain), SubDomain):
        raise Exception("No correct coupling interface defined! Given coupling domain is not of type dolfin Subdomain")

    for v in dolfin.vertices(mesh_fenics):
        if coupling_subdomain.inside(v.point(), True):
            n += 1
            fenics_vertices.append(v)
            vertices_x.append(v.x(0))
            if dimensions == 2:
                vertices_y.append(v.x(1))
            elif fenics_dimensions == 2 and dimensions == 3:
                vertices_y.append(v.x(1))
                vertices_z.append(0)
            else:
                raise Exception("Dimensions of coupling problem (dim={}) and FEniCS setup (dim={}) do not match!"
                                .format(dimensions, fenics_dimensions))

    assert (n != 0), "No coupling boundary vertices detected"

    if dimensions == 2:
        return fenics_vertices, np.stack([vertices_x, vertices_y], axis=1)
    elif dimensions == 3:
        return fenics_vertices, np.stack([vertices_x, vertices_y, vertices_z], axis=1)


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


def get_coupling_boundary_edges(mesh_fenics, coupling_subdomain, id_mapping):
    """
    Extracts edges of mesh which lie on the coupling boundary.

    Parameters
    ----------
    mesh_fenics : FEniCS Mesh
        FEniCS mesh of the complete region.
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
    vertices = dict()

    for v1 in dolfin.vertices(mesh_fenics):
        if coupling_subdomain.inside(v1.point(), True):
            vertices[v1] = []

    for v1 in vertices.keys():
        for v2 in vertices.keys():
            if are_connected_by_edge(v1, v2):
                vertices[v1] = v2
                vertices[v2] = v1

    vertices1_ids = []
    vertices2_ids = []

    for v1, v2 in vertices.items():
        if v1 is not v2:
            vertices1_ids.append(id_mapping[v1.global_index()])
            vertices2_ids.append(id_mapping[v2.global_index()])

    vertices1_ids = np.array(vertices1_ids)
    vertices2_ids = np.array(vertices2_ids)

    return vertices1_ids, vertices2_ids


def get_forces_as_point_sources(fixed_boundary, function_space, coupling_mesh_vertices, data, z_dead=False):
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
    data : PointSource
        FEniCS PointSource data carrying forces
    z_dead: bool
        Allows to ignore z dimension

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

    if z_dead:
        assert (dims == 3), "z_dead=True is only allowed for 3D data"
    else:
        assert (dims == 2), "This Adapter can create Point Sources only from 2D data. Use z_dead=True, if you want " \
                            "to ignore the z dimension."

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

