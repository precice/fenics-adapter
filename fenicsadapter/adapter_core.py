"""
This module consists of all core functionality of the FEniCS adapter .
The module also consists of additional helper functions for the user
"""

import dolfin
from dolfin import SubDomain, Point, PointSource
import numpy as np
from enum import Enum
import logging

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class InterpolationType(Enum):
    """ Defines the type of interpolation strategy used"""
    CUBIC_SPLINE = 1  # cubic splines used interpolation
    RBF = 2  # Radial basis functions used for interpolation


class FunctionType(Enum):
    """ Defines scalar- and vector-valued function """
    SCALAR = 0  # scalar valued function
    VECTOR = 1  # vector valued function


def determine_function_type(input_function):
    """ Determines if the function is scalar- or vector-valued based on
    rank evaluation.
    """
    if input_function.value_rank() == 0:  # scalar-valued functions have rank 0 is FEniCS
        return FunctionType.SCALAR
    elif input_function.value_rank() == 1:  # vector-valued functions have rank 1 in FEniCS
        return FunctionType.VECTOR
    else:
        raise Exception("Error determining function type")


def filter_point_sources(point_sources, filter_out):
    """
    Filter dictionary of PointSources (point_sources) with respect to a given domain (filter_out). If a PointSource
    is applied at a point inside of the given domain (filter_out), this PointSource will be removed from dictionary.
    :param point_sources: dictionary containing coordinates and associated PointSources;
      {(point_x, point_y): PointSource, ...}
    :param filter_out: defines the domain where PointSources should be filtered out
    :return: A dictionary with the filtered PointSources
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
    """Converts FEniCS data of type dolfin.Function into Numpy array for all x and y coordinates on the boundary.

    :param data: FEniCS boundary function
    :raise Exception: if type of data cannot be handled
    :return: array of FEniCS function values at each point on the boundary
    """
    if type(data) is dolfin.Function:
        x_all, y_all = sample_points[:, 0], sample_points[:, 1]
        return np.array([data(x, y) for x, y in zip(x_all, y_all)])
    else:
        raise Exception("Cannot handle data type %s" % type(data))


def extract_coupling_boundary_vertices(mesh_fenics, coupling_subdomain, fenics_dimensions, dimensions):
    """Extracts vertices which lie on the boundary.
    :return: stack of vertices
    """
    n = 0
    fenics_vertices = []
    vertices_x = []
    vertices_y = []
    if dimensions == 3:
        vertices_z = []

    if not issubclass(type(coupling_subdomain), SubDomain):
        raise Exception("no correct coupling interface defined!")

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
                raise Exception("Dimensions do not match!")

    assert (n != 0), "No coupling boundary vertices detected"

    if dimensions == 2:
        return fenics_vertices, np.stack([vertices_x, vertices_y], axis=1), n
    elif dimensions == 3:
        return fenics_vertices, np.stack([vertices_x, vertices_y, vertices_z], axis=1), n


def are_connected_by_edge(v1, v2):
    """Returns true if both vertices are connected by an edge. """
    for edge1 in dolfin.edges(v1):
        for edge2 in dolfin.edges(v2):
            if edge1.index() == edge2.index():  # Vertices are connected by edge
                return True
    return False


def extract_coupling_boundary_edges(mesh_fenics, coupling_subdomain, id_mapping):
    """Extracts edges of mesh which lie on the boundary.
    :return: two arrays of vertex IDs. Array 1 consists of first points of all edges
    and Array 2 consists of second points of all edges

    NOTE: Edge calculation is only relevant in 2D cases.
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


def get_forces_as_point_sources(fixed_boundary, function_space, coupling_mesh_vertices, data):
    """
    Creates 2 dicts of PointSources that can be applied to the assembled system.
    Applies filter_point_source to avoid forces being applied to already existing Dirichlet BC, since this would
    lead to an overdetermined system that cannot be solved.
    :return: Returns lists of PointSources
    """
    # PointSources are scalar valued, therefore we need an individual scalar valued PointSource for each dimension in a vector-valued setting
    # TODO: a vector valued PointSource would be more straightforward, but does not exist (as far as I know)

    x_forces = dict()  # dict of PointSources for Forces in x direction
    y_forces = dict()  # dict of PointSources for Forces in y direction

    vertices_x = coupling_mesh_vertices[:, 0]
    vertices_y = coupling_mesh_vertices[:, 1]

    n_vertices, _ = coupling_mesh_vertices.shape

    for i in range(n_vertices):
        px, py = vertices_x[i], vertices_y[i]
        key = (px, py)
        x_forces[key] = PointSource(function_space.sub(0), Point(px, py), data[i, 0])
        y_forces[key] = PointSource(function_space.sub(1), Point(px, py), data[i, 1])

    # Avoid application of PointSource and Dirichlet boundary condition at the same point by filtering
    x_forces = filter_point_sources(x_forces, fixed_boundary)
    y_forces = filter_point_sources(y_forces, fixed_boundary)

    return x_forces.values(), y_forces.values()  # don't return dictionary, but list of PointSources


def extract_coupling_boundary_coordinates(coupling_vertices, fenics_dimensions, dimensions):
    """Extracts the coordinates of vertices that lay on the boundary. 3D
    case currently handled as 2D.

    :return: x and y cooridinates.
    """
    vertices_x = coupling_vertices[:, 0]
    vertices_y = coupling_vertices[:, 1]
    if dimensions == 3:
        vertices_z = coupling_vertices[2, :]

    if dimensions == 2 or (fenics_dimensions == 2 and dimensions == 3):
        return vertices_x, vertices_y
    else:
        raise Exception("Error: These Dimensions are not supported by the adapter.")
