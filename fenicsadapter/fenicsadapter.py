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
    SCALAR = 0 # scalar valued funtion
    VECTOR = 1 # vector valued function

class CustomExpression(UserExpression):
    """Creates functional representation (for FEniCS) of nodal data
    provided by preCICE, using RBF interpolation.
    """
    def set_boundary_data(self, vals, coords_x, coords_y=None, coords_z=None):
        self.update_boundary_data(vals, coords_x, coords_y, coords_z)

    def update_boundary_data(self, vals, coords_x, coords_y=None, coords_z=None):
        self._coords_x = coords_x
        if coords_y is None:
            coords_y = np.zeros(self._coords_x.shape)
        self._coords_y = coords_y
        if coords_z is None:
            coords_z = np.zeros(self._coords_x.shape)
        self._coords_z = coords_z

        self._vals = vals

        if self.isScalarValues():
            assert (self._vals.shape == self._coords_x.shape)
        elif self.isVectorValues():
            assert (self._vals.shape[0] == self._coords_x.shape[0])

    def rbf_interpol(self, x, dim_no):
        if x.__len__() == 1: # for 1D only R->R mapping is allowed by preCICE, no need to implement Vector case
            f = Rbf(self._coords_x, self._vals.flatten())
            return f(x)
        if x.__len__() == 2:
            if self.isScalarValues(): #check if scalar or vector-valued
                f = Rbf(self._coords_x, self._coords_y, self._vals.flatten())
            elif self.isVectorValues():
                f = Rbf(self._coords_x, self._coords_y, self._vals[:,dim_no].flatten()) # extract dim_no element of each vector
            else: # TODO: this is already checked in isScalarValues()!
                raise Exception("Dimension of the function is 0 or negative!")
            return f(x[0], x[1])
        if x.__len__() == 3: # this case has not been tested yet
            if self.isScalarValues():
                f = Rbf(self._coords_x, self._coords_y, self._coords_z, self._vals.flatten())
            elif self.isVectorValues():
                f = Rbf(self._coords_x, self._coords_y, self._coords_z, self._vals[:,dim_no].flatten())
            else: # TODO: this is already checked in isScalarValues()!
                raise Exception("Dimension of the function is 0 or negative!")
            return f(x[0], x[1], x[2])

    def lin_interpol(self, x):
        f = interp1d(self._coords_x, self._vals)
        return f(x.x())

    def eval(self, value, x): # overloaded function
        """
        Overrides UserExpression.eval(). Called by Expression(x_coord). handles
        scalar- and vector-valued functions evaluation.
        """
        for i in range(self._vals.ndim):
            value[i] = self.rbf_interpol(x,i)

    def isScalarValues(self):
        """ Determines if function being interpolated is scalar-valued based on
        dimension of provided vals vector
        """
        if self._vals.ndim == 1:
            return True
        elif self._vals.ndim > 1:
            return False
        else:
            raise Exception("Dimension of the function is 0 or negative!")

    def isVectorValues(self):
        """ Determines if function being interpolated is vector-valued based on
        dimension of provided vals vector
        """
        if self._vals.ndim > 1:
            return True
        elif self._vals.ndim == 1:
            return False
        else:
            raise Exception("Dimension of the function is 0 or negative!")


class Adapter:
    """Initializes the Adapter. Initalizer creates object of class Config (from
    config.py module).

    :ivar _config: object of class Config, which stores data from the JSON config file
    """
    def __init__(self, adapter_config_filename='precice-adapter-config.json'):

        self._config = Config(adapter_config_filename)

        self._solver_name = self._config.get_solver_name()

        self._interface = precice.Interface(self._solver_name, 0, 1)
        self._interface.configure(self._config.get_config_file_name())
        self._dimensions = self._interface.get_dimensions()

        self._coupling_subdomain = None # initialized later
        self._mesh_fenics = None # initialized later
        self._coupling_bc_expression = None # initialized later
        self._fenics_dimensions = None # initialized later
        self._vector_function = None # initialized later

        ## coupling mesh related quantities
        self._coupling_mesh_vertices = None # initialized later
        self._mesh_name = self._config.get_coupling_mesh_name()
        self._mesh_id = self._interface.get_mesh_id(self._mesh_name)
        self._vertex_ids = None # initialized later
        self._n_vertices = None # initialized later

        ## write data related quantities (write data is written by this solver to preCICE)
        self._write_data_name = self._config.get_write_data_name()
        self._write_data_id = self._interface.get_data_id(self._write_data_name, self._mesh_id)
        self._write_data = None #_write_data is a vector with the values like it is used by precice (The 2D-format of values is (d0x, d0y, d1x, d1y, ..., dnx, dny) The 3D-format of values is (d0x, d0y, d0z, d1x, d1y, d1z, ..., dnx, dny, dnz))

        ## read data related quantities (read data is read by this solver from preCICE)
        self._read_data_name = self._config.get_read_data_name()
        self._read_data_id = self._interface.get_data_id(self._read_data_name, self._mesh_id)
        self._read_data = None #a numpy 1D array with the values like it is used by precice (The 2D-format of values is (d0x, d0y, d1x, d1y, ..., dnx, dny) The 3D-format of values is (d0x, d0y, d0z, d1x, d1y, d1z, ..., dnx, dny, dnz))

        ## numerics
        self._precice_tau = None

        ## checkpointing
        self._u_cp = None  # checkpoint for temperature inside domain
        self._t_cp = None  # time of the checkpoint
        self._n_cp = None  # timestep of the checkpoint

        #function space
        self._function_space = None
        self.dss = None #intgration domain for evaluation an integral over the domain
        
    def convert_fenics_to_precice(self, data, mesh, subdomain):
        """Converts FEniCS data of type dolfin.Function into
        Numpy array for all x and y coordinates on the boundary.

        :param data: FEniCS boundary function
        :raise Exception: if type of data cannot be handled
        :return: array of FEniCS function values at each point on the boundary
        """
        if type(data) is dolfin.Function:
            x_all, y_all = self.extract_coupling_boundary_coordinates()
            #TODO: Append line of zeros for 2D-3D FSI-Coupling
            return np.array([data(x, y) for x, y in zip(x_all, y_all)])
#            if self._fenics_dimensions==2 and self._dimensions==3: #Quasi 2D fenics
#                assert(precice_data.shape[1] == 2)
#                #insert zeros for y component
#                precice_data = np.vstack((precice_data[:,0], np.zeros(precice_data.shape[0]), precice_data[:,1] )).transpose()
#                assert(precice_data.shape[1] == 3)
#            return precice_data.ravel()
            #ich glaube das ist doch quatsch das hier zu machen, besser in read_block_data und write_block_data
        else:
            raise Exception("Cannot handle data type %s" % type(data))
            
    def write_block_data(self):
        """
        Writes the data to precice. Dependent on the dimensions
        of the simulation (2D-3D Coupling, 2D-2-D coupling or
        Scalar/Vector write function) write_data is first converted.
        """
        if  self._function_type is FunctionType.SCALAR: #write_field.value_rank() == 0:
                self._interface.write_block_scalar_data(self._write_data_id, self._n_vertices, self._vertex_ids, self._write_data)
        elif self._function_type is FunctionType.VECTOR:
                
                if self._fenics_dimensions == 2 and self._dimensions==3:
                    #This corresponds to 2D-3D coupling
                    #zeros have to be inserted for the y-entry
                    
                    precice_write_data = np.column_stack((self._write_data[:,0], np.zeros(self._n_vertices), self._write_data[:,1]))
                    print("Write data:")
                    print(precice_write_data.ravel())
                    assert(precice_write_data.shape[0]==self._n_vertices and precice_write_data.shape[1]==self._dimensions)                         
                    self._interface.write_block_vector_data(self._write_data_id, self._n_vertices, self._vertex_ids, precice_write_data.ravel())
                    
                elif self._fenics_dimensions == self._dimensions:
                    self._interface.write_block_vector_data(self._write_data_id, self._n_vertices, self._vertex_ids, self._write_data.ravel())

        else:
                raise Exception("Rank of function space is neither 0 nor 1")
        self._interface.fulfilled_action(precice.action_write_initial_data())

    def read_block_data(self):
        """
        Wrapper for _interface.read_block_scalar_data and _interface.read_block_vector_data.
        Checks for Scalar/vector Case and
        converts the read_data 3 x n /2 x n -array into a 1D 3*n or 2*n array.
        For quasi 2D fenics in a 3D coupled simulation the y component of the vectors is deleted.
        """
        
        if self._function_type is FunctionType.SCALAR:
            self._interface.read_block_scalar_data(self._read_data_id, self._n_vertices, self._vertex_ids, self._read_data)
        
        elif self._function_type is FunctionType.VECTOR:
                          
            if self._fenics_dimensions == self._dimensions:
                self._interface.read_block_vector_data(self._read_data_id, self._n_vertices,
                                                   self._vertex_ids, self._read_data.ravel())
                
                               
            elif self._fenics_dimensions ==2 and self._dimensions==3:
                
                precice_data = np.zeros(self._n_vertices * self._dimensions)
                self._interface.read_block_vector_data(self._read_data_id, self._n_vertices,
                                                   self._vertex_ids, precice_data)
                
                precice_read_data = np.reshape(precice_data,(self._n_vertices, self._dimensions), 'C')
                
            
                self._read_data[:,0] = precice_read_data[:,0]
                self._read_data[:,1] = precice_read_data[:,2] # y is the dead direction so it is left out
                print("read forces:", self._read_data)
            else: 
                raise Exception("Dimensions don't match!")
            
                
                
        

    def extract_coupling_boundary_vertices(self):
        """
        Extracts verticies which lay on the boundary. Currently handles 2D
        case properly, 3D is circumvented.

        :raise Exception: if no correct coupling interface is defined
        :return: stack of verticies
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
                
                elif self._dimensions == 3:
                    # todo this has to be fixed for "proper" 3D coupling. Currently this is a workaround for the coupling of 2D fenics with pseudo 3D openfoam
                    vertices_z.append(v.x(1))
                    vertices_y.append(0)

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
        self._coupling_mesh_vertices, self._n_vertices = self.extract_coupling_boundary_vertices()
        self._vertex_ids = np.zeros(self._n_vertices)
        self._interface.set_mesh_vertices(self._mesh_id, self._n_vertices, self._coupling_mesh_vertices.flatten('F'), self._vertex_ids)

    def set_write_field(self, write_function_init):
        """Sets the write field. Called by initalize() function at the
        beginning of the simulation.

        :param write_function_init: function on the write field
        """
        self._write_data = self.convert_fenics_to_precice(write_function_init, self._mesh_fenics, self._coupling_subdomain)

    def set_read_field(self, read_function_init):
        """Sets the read field. Called by initalize() function at the
        beginning of the simulation.

        :param read_function_init: function on the read field
        """
        self._read_data = self.convert_fenics_to_precice(read_function_init, self._mesh_fenics, self._coupling_subdomain)

    def create_coupling_boundary_condition(self):
        """Creates the coupling boundary conditions using CustomExpression."""
        x_vert, y_vert = self.extract_coupling_boundary_coordinates()
        try:  # works with dolfin 1.6.0
            self._coupling_bc_expression = CustomExpression(element=self._function_space.ufl_element()) # elemnt information must be provided, else DOLFIN assumes scalar function
        except (TypeError, KeyError):  # works with dolfin 2017.2.0
            self._coupling_bc_expression = CustomExpression(element=self._function_space.ufl_element(),degree=0)
        self._coupling_bc_expression.set_boundary_data(self._read_data, x_vert, y_vert)

    def create_coupling_dirichlet_boundary_condition(self, function_space):
        """Creates the coupling Dirichlet boundary conditions using
        create_coupling_boundary_condition() method.

        :return: dolfin.DirichletBC()
        """
        self._function_space = function_space
        self.create_coupling_boundary_condition()
        return dolfin.DirichletBC(self._function_space, self._coupling_bc_expression, self._coupling_subdomain)

    def create_coupling_neumann_boundary_condition(self, test_functions, boundary_marker=None):
        """Creates the coupling Neumann boundary conditions using
        create_coupling_boundary_condition() method.

        :return: expression in form of integral: g*v*ds. (see e.g. p. 83ff
         Langtangen, Hans Petter, and Anders Logg. "Solving PDEs in Python The
         FEniCS Tutorial Volume I." (2016).)
        """
        self._function_space = test_functions.function_space()
        self.create_coupling_boundary_condition()
        if not boundary_marker: #there is only 1 Neumann-BC which is at the coupling boundary -> integration over whole boundary
            
            return dot(test_functions, self._coupling_bc_expression) * dolfin.ds  # this term has to be added to weak form to add a Neumann BC (see e.g. p. 83ff Langtangen, Hans Petter, and Anders Logg. "Solving PDEs in Python The FEniCS Tutorial Volume I." (2016).)
        else: #For multiple Neumann BCs integration should only be performed over the respective domain. TODO: fix the problem here
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
        x_vert, y_vert = self.extract_coupling_boundary_coordinates()
        self._write_data = self.convert_fenics_to_precice(write_function, self._mesh_fenics, self._coupling_subdomain)

        # communication
        if self.function_type(write_function) == FunctionType.SCALAR:
            self._interface.write_block_scalar_data(self._write_data_id, self._n_vertices, self._vertex_ids, self._write_data)
        elif self.function_type(write_function) == FunctionType.VECTOR:
            self.write_block_data()
            #self._write_data[:,1] = 1
            #print(self._write_data)
            #entries = self._write_data.shape[0]
            #assert(entries == self._n_vertices)
            #dims = self._write_data.shape[1]
            #self._write_data = self._write_data.ravel()   #  reshape(entries*dims, order='C')
            #print(self._write_data)
            #self._interface.write_block_vector_data(self._write_data_id, self._n_vertices, self._vertex_ids, self._write_data)
        
        max_dt = self._interface.advance(dt)
        
        self.read_block_data()
        
#        if self.function_type(write_function) == FunctionType.SCALAR:
#            self._interface.read_block_scalar_data(self._read_data_id, self._n_vertices, self._vertex_ids, self._read_data)
#        
#        elif self.function_type(write_function) == FunctionType.VECTOR:
#            self._interface.read_block_vector_data(self._read_data_id, self._n_vertices, self._vertex_ids, self._read_data.ravel())

        # update boundary condition with read data
        self._coupling_bc_expression.update_boundary_data(self._read_data, x_vert, y_vert)
        print("Evaluate Force Field at different pionts. Should be [1 0] for Dummies")
        print(self._coupling_bc_expression((0,1)))

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
            t = new_t = t + dt  # todo the variables new_t, new_n could be saved, by just using t and n below, however I think it improved readability.
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

    def initialize(self, coupling_subdomain, mesh, read_field, write_field, 
                   u_n, dimension=2, t=0, n=0, coupling_marker=0):
        """Initializes remaining attributes. Called once, from the solver.

        :param read_field: function applied on the read field
        :param write_field: function applied on the write field
        """
        print("Begin initializating the fenics adapter...")
        
        self._fenics_dimensions = dimension
        
        self.set_coupling_mesh(mesh, coupling_subdomain)
        self.set_read_field(read_field)
        self.set_write_field(write_field)
        self._precice_tau = self._interface.initialize()
        
        self._function_type = self.function_type(write_field)
        
        if self._interface.is_action_required(precice.action_write_initial_data()):
            self.write_block_data()
                
        self._interface.initialize_data()

        if self._interface.is_read_data_available():
            self.read_block_data()
#            if self.function_type(read_field) is FunctionType.SCALAR:
#                self._interface.read_block_scalar_data(self._read_data_id, self._n_vertices, self._vertex_ids, self._read_data)
#            elif self.function_type(read_field) is FunctionType.VECTOR:
#                self._interface.read_block_vector_data(self._read_data_id, self._n_vertices, self._vertex_ids, self._read_data.ravel())
#            else:
#                raise Exception("Rank of function space is neither 0 nor 1")


        if self._interface.is_action_required(precice.action_write_iteration_checkpoint()):
            self._u_cp = u_n.copy(deepcopy=True)
            self._t_cp = t
            self._n_cp = n
            self._interface.fulfilled_action(precice.action_write_iteration_checkpoint())
            
        #create an integration domain for the coupled boundary
        self.dss=Measure('ds', domain=mesh, subdomain_data=coupling_marker)
        print("initialization successful")

        return self._precice_tau

    def is_coupling_ongoing(self):
        """Determines whether simulation should continue. Called from the
        simulation loop in the solver.

        :return: True if the coupling is ongoing, False otherwise
        """
        return self._interface.is_coupling_ongoing()

    def extract_coupling_boundary_coordinates(self):
        """Extracts the coordinates of vertices that lay on the boundary. 3D
        case currently handled as 2D.

        :return: x and y cooridinates.
        """
        vertices, _ = self.extract_coupling_boundary_vertices()
        vertices_x = vertices[0, :]
        vertices_y = vertices[1, :]
        if self._dimensions == 3:
            vertices_z = vertices[2, :]

        if self._dimensions == 2:
            return vertices_x, vertices_y
        elif self._dimensions == 3 and self._fenics_dimensions==2:
            # todo this has to be fixed for "proper" 3D coupling. Currently this is a workaround for the coupling of 2D fenics with pseudo 3D openfoam
            return vertices_x, vertices_z
        else:
            raise Exception("Error: These Dimensions are not supported by the adapter.")

    def function_type(self, function):
        """ Determines if the function is scalar- or vector-valued based on
        rank evaluation.
        """
        # TODO: is is better to use FunctionType.SCALAR.value here ?
        if function.value_rank() == 0: # scalar-valued functions have rank 0 is FEniCS
            return FunctionType.SCALAR
        elif function.value_rank() == 1: # vector-valued functions have rank 1 in FEniCS
            return FunctionType.VECTOR
        else:
            raise Exception("Error determining function type")

    def finalize(self):
        """Finalizes the coupling interface."""
        self._interface.finalize()
