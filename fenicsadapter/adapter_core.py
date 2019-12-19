"""This module consists of all core functionality of the FEniCS adapter .
   The module also consists of additional helper functions for the user
"""

import dolfin
from dolfin import UserExpression, SubDomain, Function, FacetNormal, dot, Point
from dolfin.cpp.fem import PointSource
from scipy.interpolate import Rbf
from scipy.interpolate import interp1d
import numpy as np
from .solverstate import SolverState
from enum import Enum
import logging
import precice

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


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


class CustomExpression(UserExpression):
    """Creates functional representation (for FEniCS) of nodal data
    provided by preCICE.
    """
    def set_boundary_data(self, vals, coords_x, coords_y=None, coords_z=None):
        """ initialize data stored by expression.

        :param vals: data values on nodes
        :param coords_x: x coordinates of nodes
        :param coords_y: y coordinates of nodes
        :param coords_z: z coordinates of nodes
        """
        self.update_boundary_data(vals, coords_x, coords_y, coords_z)

    def update_boundary_data(self, vals, coords_x, coords_y=None, coords_z=None):
        """ update the data stored by expression.

        :param vals: data values on nodes
        :param coords_x: x coordinates of nodes
        :param coords_y: y coordinates of nodes
        :param coords_z: z coordinates of nodes
        """
        self._coords_x = coords_x
        self._dimension = 3
        if coords_y is None:
            self._dimension -= 1
            coords_y = np.zeros(self._coords_x.shape)
        self._coords_y = coords_y
        if coords_z is None:
            self._dimension -= 1
            coords_z = np.zeros(self._coords_x.shape)

        self._coords_y = coords_y
        self._coords_z = coords_z
        self._vals = vals

        self._f = self.create_interpolant()

        if self.is_scalar_valued():
            assert (self._vals.shape == self._coords_x.shape)
        elif self.is_vector_valued():
            assert (self._vals.shape[0] == self._coords_x.shape[0])

    def interpolate(self, x):
        # TODO: the correct way to deal with this would be using an abstract class. Since this is technically more complex and the current implementation is a workaround anyway, we do not use the proper solution, but this hack.
        """ Interpolates at x. Uses buffered interpolant self._f.

        :return: returns a list containing the interpolated values. If scalar function is interpolated this list has a
        single element. If a vector function is interpolated the list has self._dimensions elements.
        """
        raise Exception("Please use one of the classes derived from this class, that implements an actual strategy for"
                        "interpolation.")

    def create_interpolant(self, x):
        # TODO: the correct way to deal with this would be using an abstract class. Since this is technically more complex and the current implementation is a workaround anyway, we do not use the proper solution, but this hack.
        """ Creates interpolant from boundary data that has been provided before.

        :return: returns interpolant as list. If scalar function is interpolated this list has a single element. If a
        vector function is interpolated the list has self._dimensions elements.
        """
        raise Exception("Please use one of the classes derived from this class, that implements an actual strategy for"
                        "interpolation.")

    def eval(self, value, x):
        """ Evaluates expression at x using self.interpolate(x) and stores result to value.

        :param x: coordinate where expression has to be evaluated
        :param value: buffer where result has to be returned to
        """
        return_value = self.interpolate(x)
        for i in range(self._vals.ndim):
            value[i] = return_value[i]

    def is_scalar_valued(self):
        """ Determines if function being interpolated is scalar-valued based on dimension of provided vector self._vals.

        :return: whether function is scalar valued
        """
        if self._vals.ndim == 1:
            return True
        elif self._vals.ndim > 1:
            return False
        else:
            raise Exception("Dimension of the function is 0 or negative!")

    def is_vector_valued(self):
        """ Determines if function being interpolated is vector-valued based on dimension of provided vector self._vals.

        :return: whether function is scalar valued
        """
        if self._vals.ndim > 1:
            return True
        elif self._vals.ndim == 1:
            return False
        else:
            raise Exception("Dimension of the function is 0 or negative!")


class GeneralInterpolationExpression(CustomExpression):
    """Uses RBF interpolation for implementation of CustomExpression.interpolate. Allows for arbitrary coupling
    interfaces, but has limited accuracy.
    """
    def create_interpolant(self):
        interpolant = []
        if self._dimension == 1:
            assert(self.is_scalar_valued())  # for 1D only R->R mapping is allowed by preCICE, no need to implement Vector case
            interpolant.append(Rbf(self._coords_x, self._vals.flatten()))
        elif self._dimension == 2:
            if self.is_scalar_valued():  # check if scalar or vector-valued
                interpolant.append(Rbf(self._coords_x, self._coords_y, self._vals.flatten()))
            elif self.is_vector_valued():
                interpolant.append(Rbf(self._coords_x, self._coords_y, self._vals[:, 0].flatten())) # extract dim_no element of each vector
                interpolant.append(Rbf(self._coords_x, self._coords_y, self._vals[:, 1].flatten())) # extract dim_no element of each vector
            else:
                raise Exception("Problem dimension and data dimension not matching.")
        elif self._dimension == 3:
            logger.warning("RBF Interpolation for 3D Simulations has not been properly tested!")
            if self.is_scalar_valued():
                interpolant.append(Rbf(self._coords_x, self._coords_y, self._coords_z, self._vals.flatten()))
            elif self.is_vector_valued():
                interpolant.append(Rbf(self._coords_x, self._coords_y, self._coords_z, self._vals[:, 0].flatten()))
                interpolant.append(Rbf(self._coords_x, self._coords_y, self._coords_z, self._vals[:, 1].flatten()))
                interpolant.append(Rbf(self._coords_x, self._coords_y, self._coords_z, self._vals[:, 2].flatten()))
            else:
                raise Exception("Problem dimension and data dimension not matching.")
        else:
            raise Exception("Dimension of the function invalid/not supported.")

        return interpolant

    def interpolate(self, x):
        assert ((self.is_scalar_valued() and self._vals.ndim == 1) or
                (self.is_vector_valued() and self._vals.ndim == self._dimension))

        return_value = self._vals.ndim * [None]

        if self._dimension == 1:
            for i in range(self._vals.ndim):
                return_value[i] = self._f[i](x[0])
        if self._dimension == 2:
            for i in range(self._vals.ndim):
                return_value[i] = self._f[i](x[0], x[1])
        if self._dimension == 3:
            for i in range(self._vals.ndim):
                return_value[i] = self._f[i](x[0], x[1], x[2])
        return return_value


class ExactInterpolationExpression(CustomExpression):
    """Uses cubic spline interpolation for implementation of CustomExpression.interpolate. Only allows interpolation on
    coupling that are parallel to the y axis, and if the coordinates in self._coords_y are ordered such that the nodes
    on the coupling mesh are traversed w.r.t their connectivity.
    However, this method allows to exactly recover the solution at the coupling interface, if it is a polynomial of
    order 3 or lower.
    See also https://github.com/precice/fenics-adapter/milestone/1
    """
    def create_interpolant(self):
        interpolant = []
        if self._dimension == 2:
            if self.is_scalar_valued():  # check if scalar or vector-valued
                interpolant.append(interp1d(self._coords_y, self._vals, bounds_error=False, fill_value="extrapolate", kind="cubic"))
            elif self.is_vector_valued():
                interpolant.append(interp1d(self._coords_y, self._vals[:, 0].flatten(), bounds_error=False, fill_value="extrapolate", kind="cubic"))
                interpolant.append(interp1d(self._coords_y, self._vals[:, 1].flatten(), bounds_error=False, fill_value="extrapolate", kind="cubic"))
            else:
                raise Exception("Problem dimension and data dimension not matching.")
        else:
            raise Exception("Dimension of the function is invalid/not supported.")

        return interpolant

    def interpolate(self, x):
        assert ((self.is_scalar_valued() and self._vals.ndim == 1) or
                (self.is_vector_valued() and self._vals.ndim == self._dimension))

        return_value = self._vals.ndim * [None]

        if self._dimension == 2:
            for i in range(self._vals.ndim):
                return_value[i] = self._f[i](x[1])
        else:
            raise Exception("invalid dimensionality!")
        return return_value


def convert_fenics_to_precice(self, data):
    """Converts FEniCS data of type dolfin.Function into Numpy array for all x and y coordinates on the boundary.

    :param data: FEniCS boundary function
    :raise Exception: if type of data cannot be handled
    :return: array of FEniCS function values at each point on the boundary
    """
    if type(data) is dolfin.Function:
        x_all, y_all = self._extract_coupling_boundary_coordinates()
        return np.array([data(x, y) for x, y in zip(x_all, y_all)])
    else:
        raise Exception("Cannot handle data type %s" % type(data))


def extract_coupling_boundary_vertices(self):
    """Extracts vertices which lie on the boundary.
    :return: stack of vertices
    """
    n = 0
    fenics_vertices = []
    vertices_x = []
    vertices_y = []
    if self._dimensions == 3:
        vertices_z = []

    if not issubclass(type(self._coupling_subdomain), SubDomain):
        raise Exception("no correct coupling interface defined!")

    for v in dolfin.vertices(self._mesh_fenics):
        if self._coupling_subdomain.inside(v.point(), True):
            n += 1
            fenics_vertices.append(v)
            vertices_x.append(v.x(0))
            if self._dimensions == 2:
                vertices_y.append(v.x(1))
            elif self._can_apply_2d_3d_coupling():
                vertices_y.append(v.x(1))
                vertices_z.append(0)
            else:
                raise Exception("Dimensions do not match!")

    assert (n != 0), "No coupling boundary vertices detected"

    if self._dimensions == 2:
        return fenics_vertices, np.stack([vertices_x, vertices_y], axis=1), n
    elif self._dimensions == 3:
        return fenics_vertices, np.stack([vertices_x, vertices_y, vertices_z], axis=1), n


def are_connected_by_edge(v1, v2):
    """Returns true if both vertices are connected by an edge. """
    for edge1 in dolfin.edges(v1):
        for edge2 in dolfin.edges(v2):
            if edge1.index() == edge2.index():  # Vertices are connected by edge
                return True
    return False


def extract_coupling_boundary_edges(self, id_mapping):
    """Extracts edges of mesh which lie on the boundary.
    :return: two arrays of vertex IDs. Array 1 consists of first points of all edges
    and Array 2 consists of second points of all edges

    NOTE: Edge calculation is only relevant in 2D cases.
    """

    vertices = dict()

    for v1 in dolfin.vertices(self._mesh_fenics):
        if self._coupling_subdomain.inside(v1.point(), True):
            vertices[v1] = []

    for v1 in vertices.keys():
        for v2 in vertices.keys():
            if self._are_connected_by_edge(v1, v2):
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


def set_write_field(self, write_function_init):
    """Sets the write field. Called by initalize() function at the
    beginning of the simulation.

    :param write_function_init: function on the write field
    """
    self._write_function_type = determine_function_type(write_function_init)
    self._write_data = self._convert_fenics_to_precice(write_function_init)


def set_read_field(self, read_function_init):
    """Sets the read field. Called by initalize() function at the
    beginning of the simulation.

    :param read_function_init: function on the read field
    """
    self._read_function_type = determine_function_type(read_function_init)
    self._read_data = self._convert_fenics_to_precice(read_function_init)


def create_coupling_boundary_condition(self):
    """Creates the coupling boundary conditions using an actual implementation of CustomExpression."""
    x_vert, y_vert = self._extract_coupling_boundary_coordinates()

    try:  # works with dolfin 1.6.0
        self._coupling_bc_expression = self._my_expression(
            element=self._function_space.ufl_element())  # elemnt information must be provided, else DOLFIN assumes scalar function
    except (TypeError, KeyError):  # works with dolfin 2017.2.0
        self._coupling_bc_expression = self._my_expression(element=self._function_space.ufl_element(), degree=0)
    self._coupling_bc_expression.set_boundary_data(self._read_data, x_vert, y_vert)


def create_coupling_dirichlet_boundary_condition(self, function_space):
    """Creates the coupling Dirichlet boundary conditions using
    create_coupling_boundary_condition() method.

    :return: dolfin.DirichletBC()
    """
    self._function_space = function_space
    self._create_coupling_boundary_condition()
    return dolfin.DirichletBC(self._function_space, self._coupling_bc_expression, self._coupling_subdomain)


def create_coupling_neumann_boundary_condition(self, test_functions, function_space=None, boundary_marker=None):
    """Creates the coupling Neumann boundary conditions using
    create_coupling_boundary_condition() method.

    :return: expression in form of integral: g*v*ds. (see e.g. p. 83ff
     Langtangen, Hans Petter, and Anders Logg. "Solving PDEs in Python The
     FEniCS Tutorial Volume I." (2016).)
    """
    if not function_space:
        self._function_space = test_functions.function_space()
    else:
        self._function_space = function_space
    self._create_coupling_boundary_condition()
    if not boundary_marker:  # there is only 1 Neumann-BC which is at the coupling boundary -> integration over whole boundary
        if self._coupling_bc_expression.is_scalar_valued():
            return test_functions * self._coupling_bc_expression * dolfin.ds  # this term has to be added to weak form to add a Neumann BC (see e.g. p. 83ff Langtangen, Hans Petter, and Anders Logg. "Solving PDEs in Python The FEniCS Tutorial Volume I." (2016).)
        elif self._coupling_bc_expression.is_vector_valued():
            n = FacetNormal(self._mesh_fenics)
            return -test_functions * dot(n, self._coupling_bc_expression) * dolfin.ds
        else:
            raise Exception("invalid!")
    else:  # For multiple Neumann BCs integration should only be performed over the respective domain.
        # TODO: fix the problem here
        raise Exception("Boundary markers are not implemented yet")
        return dot(self._coupling_bc_expression, test_functions) * self.dss(boundary_marker)


def create_force_boundary_condition(self, function_space):
    """
    Initializes force-coupling via PointSource.

    This function only works for 2D-pseudo3D coupling.

    :param function_space: The Function Space used for the Test and Trial functions
    """
    self._function_space = function_space
    self._has_force_boundary = True

    return self._get_forces_as_point_sources()


def get_forces_as_point_sources(self):
    """
    Creates 2 dicts of PointSources that can be applied to the assembled system.
    Applies filter_point_source to avoid forces being applied to already existing Dirichlet BC, since this would
    lead to an overdetermined system that cannot be solved.
    :return: Returns lists of PointSources TODO: get rid of this legacy code, dicts should be used for a PointSource, since they can provide the location of the PointSouce, as well. Even, inside the FEniCS user code.
    """
    # PointSources are scalar valued, therefore we need an individual scalar valued PointSource for each dimension in a vector-valued setting
    # TODO: a vector valued PointSource would be more straightforward, but does not exist (as far as I know)

    x_forces = dict()  # dict of PointSources for Forces in x direction
    y_forces = dict()  # dict of PointSources for Forces in y direction

    vertices_x = self._coupling_mesh_vertices[:, 0]
    vertices_y = self._coupling_mesh_vertices[:, 1]

    for i in range(self._n_vertices):
        px, py = vertices_x[i], vertices_y[i]
        key = (px, py)
        x_forces[key] = PointSource(self._function_space.sub(0),
                                    Point(px, py),
                                    self._read_data[i, 0])
        y_forces[key] = PointSource(self._function_space.sub(1),
                                    Point(px, py),
                                    self._read_data[i, 1])

    # Avoid application of PointSource and Dirichlet boundary condition at the same point by filtering
    x_forces = filter_point_sources(x_forces, self._Dirichlet_Boundary)
    y_forces = filter_point_sources(y_forces, self._Dirichlet_Boundary)

    return x_forces.values(), y_forces.values()  # don't return dictionary, but list of PointSources


def can_apply_2d_3d_coupling(self):
    """ In certain situations a 2D-3D coupling is applied. This means that the y-dimension of data and nodes
    received from preCICE is ignored. If FEniCS sends data to preCICE, the y-dimension of data and node coordinates
    is set to zero.

    :return: True, if the 2D-3D coupling can be applied
    """
    return self._fenics_dimensions == 2 and self._dimensions == 3


def extract_coupling_boundary_coordinates(self):
    """Extracts the coordinates of vertices that lay on the boundary. 3D
    case currently handled as 2D.

    :return: x and y cooridinates.
    """
    _, vertices, _ = self._extract_coupling_boundary_vertices()
    vertices_x = vertices[:, 0]
    vertices_y = vertices[:, 1]
    if self._dimensions == 3:
        vertices_z = vertices[2, :]

    if self._dimensions == 2 or self._can_apply_2d_3d_coupling():
        return vertices_x, vertices_y
    else:
        raise Exception("Error: These Dimensions are not supported by the adapter.")


"""
old advance function which wrapped all preCICE functionality which is now named ---> old_advance 
"""

def old_advance(self, write_function, u_np1, u_n, t, dt, n):
    """Calls preCICE advance function using precice and manages checkpointing.
    The solution u_n is updated by this function via call-by-reference. The corresponding values for t and n are returned.

    This means:
    * either, the old value of the checkpoint is assigned to u_n to repeat the iteration,
    * or u_n+1 is assigned to u_n and the checkpoint is updated correspondingly.

    :param write_function: a FEniCS function being sent to the other participant as boundary condition at the coupling interface
    :param u_np1: new value of FEniCS solution u_n+1 at time t_n+1 = t+dt
    :param u_n: old value of FEniCS solution u_n at time t_n = t; updated via call-by-reference
    :param t: current time t_n for timestep n
    :param dt: timestep size dt = t_n+1 - t_n
    :param n: current timestep
    :return: return starting time t and timestep n for next FEniCS solver iteration. u_n is updated by advance correspondingly.
    """

    state = SolverState(u_n, t, n)

    # sample write data at interface
    x_vert, y_vert = self._extract_coupling_boundary_coordinates()
    self._write_data = self._convert_fenics_to_precice(write_function)

    # communication
    self._write_block_data()

    max_dt = self._interface.advance(dt)

    self._read_block_data()

    # update boundary condition with read data
    if self._has_force_boundary:
        x_forces, y_forces = self._get_forces_as_point_sources()
    else:
        self._coupling_bc_expression.update_boundary_data(self._read_data, x_vert, y_vert)

    solver_state_has_been_restored = False

    # checkpointing
    if self._interface.is_action_required(precice.action_read_iteration_checkpoint()):
        assert (not self._interface.is_timestep_complete())  # avoids invalid control flow
        self._restore_solver_state_from_checkpoint(state)
        solver_state_has_been_restored = True
    else:
        self._advance_solver_state(state, u_np1, dt)

    if self._interface.is_action_required(precice.action_write_iteration_checkpoint()):
        assert (not solver_state_has_been_restored)  # avoids invalid control flow
        assert (self._interface.is_timestep_complete())  # avoids invalid control flow
        self._save_solver_state_to_checkpoint(state)

    precice_step_complete = self._interface.is_timestep_complete()

    _, t, n = state.get_state()
    # TODO: this if-else statement smells.
    if self._has_force_boundary:
        return t, n, precice_step_complete, max_dt, x_forces, y_forces
    else:
        return t, n, precice_step_complete, max_dt

