"""This module handles CustomExpression and initalization of the FEniCS
adapter.

:raise ImportError: if PRECICE_ROOT is not defined
"""
import dolfin
from dolfin import UserExpression, SubDomain, Function, Measure, Expression, dot
from scipy.interpolate import Rbf
from scipy.interpolate import interp1d
import numpy as np
from .config import Config
from enum import Enum
import logging

try:
    import precice
except ImportError:
    import os
    import sys
    # check if PRECICE_ROOT is defined
    if not os.getenv('PRECICE_ROOT'):
       raise Exception("ERROR: PRECICE_ROOT not defined!")

    precice_root = os.getenv('PRECICE_ROOT')
    precice_python_adapter_root = precice_root+"/src/precice/bindings/python"
    sys.path.insert(0, precice_python_adapter_root)
    import precice


class FunctionType(Enum):
    """ Defines scalar- and vector-valued function """
    SCALAR = 0  # scalar valued function
    VECTOR = 1  # vector valued function


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
            logging.warning("RBF Interpolation for 3D Simulations has not been properly tested!")
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
    """Uses cubic spline interpolation for implementation of CustomExpression.interpolate. Only allows intepolation on
    coupling that are parallel to the y axis, and if the coordinates in self._coords_y are ordered such that the nodes
    on the coupling mesh are traversed w.r.t their connectivity.
    However, this method allows to exactly recover the solution at the coupling interface, if it is a polynomial of
    order 3 or lower.
    See also https://github.com/precice/fenics-adapter/milestone/1
    """
    def create_interpolant(self):
        interpolant = []
        if self._dimension == 2:
            assert(self.is_scalar_valued())  # for 1D only R->R mapping is allowed by preCICE, no need to implement Vector case
            interpolant.append(interp1d(self._coords_y, self._vals, bounds_error=False, fill_value="extrapolate", kind="cubic"))
        elif self.is_vector_valued():
            raise Exception("Vector valued functions are not supported by ExactInterpolationExpression. "
                            "Use GeneralInterpolationExpression.")
        else:
            raise Exception("Dimension of the function is invalid/not supported.")

        return interpolant

    def interpolate(self, x):
        assert (self.is_scalar_valued() and self._vals.ndim == 1)
        return [self._f[0](x[1])]


class Adapter:
    """Initializes the Adapter. Initalizer creates object of class Config (from
    config.py module).

    :ivar _config: object of class Config, which stores data from the JSON config file
    """
    def __init__(self, adapter_config_filename='precice-adapter-config.json',
                 interpolation_strategy=GeneralInterpolationExpression):

        self._config = Config(adapter_config_filename)

        self._solver_name = self._config.get_solver_name()

        self._interface = precice.Interface(self._solver_name, 0, 1)
        self._interface.configure(self._config.get_config_file_name())
        self._dimensions = self._interface.get_dimensions()

        self._coupling_subdomain = None  # initialized later
        self._mesh_fenics = None  # initialized later
        self._coupling_bc_expression = None  # initialized later
        self._fenics_dimensions = None  # initialized later

        # coupling mesh related quantities
        self._coupling_mesh_vertices = None  # initialized later
        self._mesh_name = self._config.get_coupling_mesh_name()
        self._mesh_id = self._interface.get_mesh_id(self._mesh_name)
        self._vertex_ids = None  # initialized later
        self._n_vertices = None  # initialized later

        # write data related quantities (write data is written by this solver to preCICE)
        self._write_data_name = self._config.get_write_data_name()
        self._write_data_id = self._interface.get_data_id(self._write_data_name, self._mesh_id)
        self._write_data = None  # a numpy 1D array with the values like it is used by precice (The 2D-format of values is (d0x, d0y, d1x, d1y, ..., dnx, dny) The 3D-format of values is (d0x, d0y, d0z, d1x, d1y, d1z, ..., dnx, dny, dnz))
        self._write_function_type = None  # stores whether write function is scalar or vector valued

        # read data related quantities (read data is read by this solver from preCICE)
        self._read_data_name = self._config.get_read_data_name()
        self._read_data_id = self._interface.get_data_id(self._read_data_name, self._mesh_id)
        self._read_data = None  # a numpy 1D array with the values like it is used by precice (The 2D-format of values is (d0x, d0y, d1x, d1y, ..., dnx, dny) The 3D-format of values is (d0x, d0y, d0z, d1x, d1y, d1z, ..., dnx, dny, dnz))
        self._write_function_type = None  # stores whether read function is scalar or vector valued

        # numerics
        self._precice_tau = None
        self._my_expression = interpolation_strategy

        # checkpointing
        self._u_cp = None  # checkpoint for temperature inside domain
        self._t_cp = None  # time of the checkpoint
        self._n_cp = None  # timestep of the checkpoint

        # function space
        self._function_space = None
        self._dss = None  # measure for boundary integral
        
    def _convert_fenics_to_precice(self, data):
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
            
    def _write_block_data(self):
        """ Writes data to preCICE. Depending on the dimensions of the simulation (2D-3D Coupling, 2D-2D coupling or
        Scalar/Vector write function) write_data is first converted.
        """

        assert (self._write_function_type in list(FunctionType))

        if self._write_function_type is FunctionType.SCALAR:
                self._interface.write_block_scalar_data(self._write_data_id, self._n_vertices, self._vertex_ids,
                                                        self._write_data)
        elif self._write_function_type is FunctionType.VECTOR:
                
                if self._can_apply_2d_3d_coupling():
                    # in 2d-3d coupling z dimension is set to zero
                    precice_write_data = np.column_stack((self._write_data[:, 0],
                                                          self._write_data[:, 1],
                                                          np.zeros(self._n_vertices)))

                    assert(precice_write_data.shape[0] == self._n_vertices and
                           precice_write_data.shape[1] == self._dimensions)

                    self._interface.write_block_vector_data(self._write_data_id, self._n_vertices, self._vertex_ids,
                                                            precice_write_data.ravel())
                    
                elif self._fenics_dimensions == self._dimensions:
                    self._interface.write_block_vector_data(self._write_data_id, self._n_vertices, self._vertex_ids,
                                                            self._write_data.ravel())
                else:
                    raise Exception("Dimensions don't match.")
        else:
            raise Exception("Rank of function space is neither 0 nor 1")

    def _read_block_data(self):
        """ Reads data from preCICE. Depending on the dimensions of the simulation (2D-3D Coupling, 2D-2D coupling or
        Scalar/Vector write function) read_data is converted.

        Note: For quasi 2D fenics in a 3D coupled simulation the y component of the vectors is deleted.
        """

        assert(self._read_function_type in list(FunctionType))
        
        if self._read_function_type is FunctionType.SCALAR:
            self._interface.read_block_scalar_data(self._read_data_id, self._n_vertices, self._vertex_ids,
                                                   self._read_data)
        
        elif self._read_function_type is FunctionType.VECTOR:
            if self._fenics_dimensions == self._dimensions:
                self._interface.read_block_vector_data(self._read_data_id, self._n_vertices, self._vertex_ids,
                                                       self._read_data.ravel())
                               
            elif self._can_apply_2d_3d_coupling():
                precice_data = np.zeros(self._n_vertices * self._dimensions)
                self._interface.read_block_vector_data(self._read_data_id, self._n_vertices, self._vertex_ids,
                                                       precice_data)
                
                precice_read_data = np.reshape(precice_data,(self._n_vertices, self._dimensions), 'C')

                self._read_data[:, 0] = precice_read_data[:, 0]
                self._read_data[:, 1] = precice_read_data[:, 1]
                # z is the dead direction so it is supposed that the data is close to zero
                assert(np.testing.assert_allclose(precice_read_data[:, 2], np.zeros_like(precice_read_data[:, 2])))
            else: 
                raise Exception("Dimensions don't match.")
        else:
            raise Exception("Rank of function space is neither 0 nor 1")

    def _extract_coupling_boundary_vertices(self):
        """Extracts vertices which lie on the boundary.
        :return: stack of vertices
        """
        n = 0
        vertices_x = []
        vertices_y = []
        if self._dimensions == 3:
            vertices_z = []

        if not issubclass(type(self._coupling_subdomain), SubDomain):
            raise Exception("no correct coupling interface defined!")

        for v in dolfin.vertices(self._mesh_fenics):
            if self._coupling_subdomain.inside(v.point(), True):
                n += 1
                vertices_x.append(v.x(0))
                if self._dimensions == 2:
                    vertices_y.append(v.x(1))
                elif self._can_apply_2d_3d_coupling():
                    vertices_y.append(v.x(1))
                    vertices_z.append(0)
                else:
                    raise Exception("Dimensions do not match!")

        if self._dimensions == 2:
            return np.stack([vertices_x, vertices_y]), n
        elif self._dimensions == 3:
            return np.stack([vertices_x, vertices_y, vertices_z]), n

    def set_coupling_mesh(self, mesh, subdomain):
        """Sets the coupling mesh. Called by initalize() function at the
        beginning of the simulation.
        """
        self._coupling_subdomain = subdomain
        self._mesh_fenics = mesh
        self._coupling_mesh_vertices, self._n_vertices = self._extract_coupling_boundary_vertices()
        self._vertex_ids = np.zeros(self._n_vertices)
        self._interface.set_mesh_vertices(self._mesh_id, self._n_vertices, self._coupling_mesh_vertices.flatten('F'), self._vertex_ids)

    def _set_write_field(self, write_function_init):
        """Sets the write field. Called by initalize() function at the
        beginning of the simulation.

        :param write_function_init: function on the write field
        """
        self._write_function_type = self._determine_function_type(write_function_init)
        self._write_data = self._convert_fenics_to_precice(write_function_init)

    def _set_read_field(self, read_function_init):
        """Sets the read field. Called by initalize() function at the
        beginning of the simulation.

        :param read_function_init: function on the read field
        """
        self._read_function_type = self._determine_function_type(read_function_init)
        self._read_data = self._convert_fenics_to_precice(read_function_init)

    def _create_coupling_boundary_condition(self):
        """Creates the coupling boundary conditions using an actual implementation of CustomExpression."""
        x_vert, y_vert = self._extract_coupling_boundary_coordinates()

        try:  # works with dolfin 1.6.0
            self._coupling_bc_expression = self._my_expression(element=self._function_space.ufl_element()) # elemnt information must be provided, else DOLFIN assumes scalar function
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

    def create_coupling_neumann_boundary_condition(self, test_functions, boundary_marker=None):
        """Creates the coupling Neumann boundary conditions using
        create_coupling_boundary_condition() method.

        :return: expression in form of integral: g*v*ds. (see e.g. p. 83ff
         Langtangen, Hans Petter, and Anders Logg. "Solving PDEs in Python The
         FEniCS Tutorial Volume I." (2016).)
        """
        self._function_space = test_functions.function_space()
        self._create_coupling_boundary_condition()
        if not boundary_marker: # there is only 1 Neumann-BC which is at the coupling boundary -> integration over whole boundary
            return dot(test_functions, self._coupling_bc_expression) * dolfin.ds  # this term has to be added to weak form to add a Neumann BC (see e.g. p. 83ff Langtangen, Hans Petter, and Anders Logg. "Solving PDEs in Python The FEniCS Tutorial Volume I." (2016).)
        else: # For multiple Neumann BCs integration should only be performed over the respective domain.
            # TODO: fix the problem here
            raise Exception("Boundary markers are not implemented yet")
            return dot(self._coupling_bc_expression, test_functions) * self.dss(boundary_marker)

    def advance(self, write_function, u_np1, u_n, t, dt, n):
        """Calls preCICE advance function using precice and manages checkpointing.
        The solution u_n is updated by this function via call-by-reference. The corresponding values for t and n are returned.

        This means:
        * either, the checkpoint self._u_cp is assigned to u_n to repeat the iteration,
        * or u_n+1 is assigned to u_n and the checkpoint is updated correspondingly.

        :param write_function: a FEniCS function being sent to the other participant as boundary condition at the coupling interface
        :param u_np1: new value of FEniCS solution u_n+1 at time t_n+1 = t+dt
        :param u_n: old value of FEniCS solution u_n at time t_n = t; updated via call-by-reference
        :param t: current time t_n for timestep n
        :param dt: timestep size dt = t_n+1 - t_n
        :param n: current timestep
        :return: return starting time t and timestep n for next FEniCS solver iteration. u_n is updated by advance correspondingly.
        """

        # sample write data at interface
        x_vert, y_vert = self._extract_coupling_boundary_coordinates()
        self._write_data = self._convert_fenics_to_precice(write_function)

        # communication
        self._write_block_data()

        max_dt = self._interface.advance(dt)
        
        self._read_block_data()
        
        # update boundary condition with read data
        self._coupling_bc_expression.update_boundary_data(self._read_data, x_vert, y_vert)

        precice_step_complete = False

        # checkpointing
        if self._interface.is_action_required(precice.action_read_iteration_checkpoint()):
            # continue FEniCS computation from checkpoint
            u_n.assign(self._u_cp)  # set u_n to value of checkpoint
            t = self._t_cp
            n = self._n_cp
            self._interface.fulfilled_action(precice.action_read_iteration_checkpoint())
        else:
            u_n.assign(u_np1)
            t = new_t = t + dt  # TODO: the variables new_t, new_n could be saved, by just using t and n below, however I think it improved readability.
            n = new_n = n + 1

        if self._interface.is_action_required(precice.action_write_iteration_checkpoint()):
            # continue FEniCS computation with u_np1
            # update checkpoint
            self._u_cp.assign(u_np1)
            self._t_cp = new_t
            self._n_cp = new_n
            self._interface.fulfilled_action(precice.action_write_iteration_checkpoint())
            precice_step_complete = True

        return t, n, precice_step_complete, max_dt

    def _can_apply_2d_3d_coupling(self):
        """ In certain situations a 2D-3D coupling is applied. This means that the y-dimension of data and nodes
        received from preCICE is ignored. If FEniCS sends data to preCICE, the y-dimension of data and node coordinates
        is set to zero.

        :return: True, if the 2D-3D coupling can be applied
        """
        return self._fenics_dimensions == 2 and self._dimensions == 3

    def initialize(self, coupling_subdomain, mesh, read_field, write_field, 
                   u_n, dimension=2, t=0, n=0, coupling_marker=0):
        """Initializes remaining attributes. Called once, from the solver.

        :param coupling_subdomain: domain where coupling takes place
        :param mesh: fenics mesh
        :param read_field: function applied on the read field
        :param write_field: function applied on the write field
        :param u_n: initial data for solution
        :param dimension: problem dimension
        :param t: starting time
        :param n: time step n
        :param coupling_marker: boundary marker, can be used for coupling, multiple Neumann boundary conditions are applied
        """
        self._fenics_dimensions = dimension

        if self._fenics_dimensions != self._dimensions:
            logging.warning("fenics_dimension = {} and precice_dimension = {} do not match!".format(self._fenics_dimensions,
                                                                                            self._dimensions))
            if self._can_apply_2d_3d_coupling():
                logging.warning("2D-3D coupling will be applied. Y coordinates of all nodes are set to zero.")
            else:
                raise Exception("fenics_dimension = {}, precice_dimension = {}. "
                                "No proper treatment for dimensional mismatch is implemented. Aborting!".format(
                    self._fenics_dimensions,
                    self._dimensions))

        self.set_coupling_mesh(mesh, coupling_subdomain)
        self._set_read_field(read_field)
        self._set_write_field(write_field)
        self._precice_tau = self._interface.initialize()
        
        if self._interface.is_action_required(precice.action_write_initial_data()):
            self._write_block_data()
            self._interface.fulfilled_action(precice.action_write_initial_data())
                
        self._interface.initialize_data()

        if self._interface.is_read_data_available():
            self._read_block_data()

        self._interface.initialize_data()

        if self._interface.is_action_required(precice.action_write_iteration_checkpoint()):
            self._u_cp = u_n.copy(deepcopy=True)
            self._t_cp = t
            self._n_cp = n
            self._interface.fulfilled_action(precice.action_write_iteration_checkpoint())
            
        self._dss = Measure('ds', domain=mesh, subdomain_data=coupling_marker)
        return self._precice_tau

    def is_coupling_ongoing(self):
        """Determines whether simulation should continue. Called from the
        simulation loop in the solver.

        :return: True if the coupling is ongoing, False otherwise
        """
        return self._interface.is_coupling_ongoing()

    def _extract_coupling_boundary_coordinates(self):
        """Extracts the coordinates of vertices that lay on the boundary. 3D
        case currently handled as 2D.

        :return: x and y cooridinates.
        """
        vertices, _ = self._extract_coupling_boundary_vertices()
        vertices_x = vertices[0, :]
        vertices_y = vertices[1, :]
        if self._dimensions == 3:
            vertices_z = vertices[2, :]

        if self._dimensions == 2 or self._can_apply_2d_3d_coupling():
            return vertices_x, vertices_y
        else:
            raise Exception("Error: These Dimensions are not supported by the adapter.")

    def _determine_function_type(self, function):
        """ Determines if the function is scalar- or vector-valued based on
        rank evaluation.
        """
        if function.value_rank() == 0:  # scalar-valued functions have rank 0 is FEniCS
            return FunctionType.SCALAR
        elif function.value_rank() == 1:  # vector-valued functions have rank 1 in FEniCS
            return FunctionType.VECTOR
        else:
            raise Exception("Error determining function type")

    def finalize(self):
        """Finalizes the coupling interface."""
        self._interface.finalize()

    def get_solver_name(self):
        """Returns name of this solver as defined in config file.
        :return: Solver name.
        """
        return self._solver_name
