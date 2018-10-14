import dolfin
from dolfin import Expression, SubDomain
from scipy.interpolate import Rbf
from scipy.interpolate import interp1d
import numpy as np
try:
    import PySolverInterface
except ImportError:
    import os
    import sys
    # check if PRECICE_ROOT is defined
    if not os.getenv('PRECICE_ROOT'):
       raise Exception("ERROR: PRECICE_ROOT not defined!")

    precice_root = os.getenv('PRECICE_ROOT')
    precice_python_adapter_root = precice_root+"/src/precice/bindings/python"
    sys.path.insert(0, precice_python_adapter_root)
    import PySolverInterface


class CustomExpression(Expression):
    def __init__(self, vals, coords_x, coords_y=None, coords_z=None, *args, **kwargs):
        self.update_boundary_data(vals, coords_x, coords_y, coords_z)

    def update_boundary_data(self, vals, coords_x, coords_y=None, coords_z=None):
        self._coords_x = coords_x
        if coords_y is None:
            coords_y = np.zeros(self._coords_x.shape)
        self._coords_y = coords_y
        if coords_z is None:
            coords_z = np.zeros(self._coords_x.shape)
        self._coords_z = coords_z

        self._vals = vals.flatten()
        assert (self._vals.shape == self._coords_x.shape)

    def rbf_interpol(self, x):
        if x.__len__() == 1:
            f = Rbf(self._coords_x, self._vals.flatten())
            return f(x)
        if x.__len__() == 2:
            f = Rbf(self._coords_x, self._coords_y, self._vals.flatten())
            return f(x[0], x[1])
        if x.__len__() == 3:
            f = Rbf(self._coords_x, self._coords_y, self._coords_z, self._vals.flatten())
            return f(x[0], x[1], x[2])

    def lin_interpol(self, x):
        f = interp1d(self._coords_x, self._vals)
        return f(x.x())

    def eval(self, value, x):
        value[0] = self.rbf_interpol(x)


class Coupling(object):
    def __init__(self, config_file_name, solver_name):
        self._solverName = solver_name  # name of the solver, like defined in the config

        self._interface = PySolverInterface.PySolverInterface(solver_name, 0, 1)
        self._interface.configure(config_file_name)
        self._dimensions = self._interface.getDimensions()

        self._coupling_subdomain = None  # FEniCS subdomain defining the coupling interface
        self._mesh_fenics = None  # FEniCS mesh where the coupled simulation takes place
        self._coupling_bc_expression = None  # Expression describing the coupling condition

        ## coupling mesh related quantities will be defined later
        self._coupling_mesh_vertices = None  # mesh vertices in a format that can be understood by preCICE
        self._mesh_id = None  # ID of the coupling mesh created from mesh name
        self._vertex_ids = None  # ID of vertices, will be filled by preCICE
        self._n_vertices = None  # number of vertices

        ## write data related quantities will be defined later (write data is written by this solver to preCICE)
        self._write_data_id = None  # ID of the data on the coupling mesh created from data name
        self._write_data = None  # actual data

        ## read data related quantities will be defined later (read data is read by this solver from preCICE)
        self._read_data_id = None  # ID of the data on the coupling mesh created from data name
        self._read_data = None  # actual data

        ## numerics
        self._precice_tau = None

        print("Done setting up coupling for participant with name %s." % solver_name)

    def convert_fenics_to_precice(self, data, mesh, subdomain):
        if type(data) is dolfin.Function:
            x_all, y_all = self.extract_coupling_boundary_coordinates()
            return np.array([data(x, y) for x, y in zip(x_all, y_all)])
        else:
            raise Exception("Cannot handle data type %s" % type(data))

    def extract_coupling_boundary_vertices(self):
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
                vertices_y.append(v.x(1))
                if self._dimensions == 3:
                    # todo this has to be fixed for "proper" 3D coupling. Currently this is a workaround for the coupling of 2D fenics with pseudo 3D openfoam
                    vertices_z.append(0)

        if self._dimensions == 2:
            return np.stack([vertices_x, vertices_y]), n
        elif self._dimensions == 3:
            return np.stack([vertices_x, vertices_y, vertices_z]), n

    def set_coupling_mesh(self, mesh, subdomain, coupling_mesh_name):
        self._coupling_subdomain = subdomain
        self._mesh_fenics = mesh
        self._coupling_mesh_vertices, self._n_vertices = self.extract_coupling_boundary_vertices()
        self._mesh_id = self._interface.getMeshID(coupling_mesh_name)
        self._vertex_ids = np.zeros(self._n_vertices)
        self._interface.setMeshVertices(self._mesh_id, self._n_vertices, self._coupling_mesh_vertices.flatten('F'), self._vertex_ids)

    def set_write_field(self, write_data_name, write_function_init):
        self._write_data_id = self._interface.getDataID(write_data_name, self._mesh_id)
        self._write_data = self.convert_fenics_to_precice(write_function_init, self._mesh_fenics, self._coupling_subdomain)

    def set_read_field(self, read_data_name, read_function_init):
        self._read_data_id = self._interface.getDataID(read_data_name, self._mesh_id)
        self._read_data = self.convert_fenics_to_precice(read_function_init, self._mesh_fenics, self._coupling_subdomain)

    def create_coupling_boundary_condition(self):
        x_vert, y_vert = self.extract_coupling_boundary_coordinates()

        try:  # works with dolfin 1.6.0
            self._coupling_bc_expression = CustomExpression(self._read_data, x_vert, y_vert)
        except (TypeError, KeyError):  # works with dolfin 2017.2.0
            self._coupling_bc_expression = CustomExpression(self._read_data, x_vert, y_vert, degree=0)

    def create_coupling_dirichlet_boundary_condition(self, function_space):
        self.create_coupling_boundary_condition()
        return dolfin.DirichletBC(function_space, self._coupling_bc_expression, self._coupling_subdomain)

    def create_coupling_neumann_boundary_condition(self, test_functions):
        self.create_coupling_boundary_condition()
        return self._coupling_bc_expression * test_functions * dolfin.ds  # this term has to be added to weak form to add a Neumann BC (see e.g. p. 83ff Langtangen, Hans Petter, and Anders Logg. "Solving PDEs in Python The FEniCS Tutorial Volume I." (2016).)

    def exchange_data(self, write_function, dt):
        x_vert, y_vert = self.extract_coupling_boundary_coordinates()
        self._write_data = self.convert_fenics_to_precice(write_function, self._mesh_fenics, self._coupling_subdomain)
        self._interface.writeBlockScalarData(self._write_data_id, self._n_vertices, self._vertex_ids, self._write_data)
        self._interface.advance(dt)
        self._interface.readBlockScalarData(self._read_data_id, self._n_vertices, self._vertex_ids, self._read_data)
        self._coupling_bc_expression.update_boundary_data(self._read_data, x_vert, y_vert)

    def initialize_data(self):
        self._precice_tau = self._interface.initialize()

        if self._interface.isActionRequired(PySolverInterface.PyActionWriteInitialData()):
            self._interface.writeBlockScalarData(self._read_data_id, self._n_vertices, self._vertex_ids, self._read_data)
            self._interface.fulfilledAction(PySolverInterface.PyActionWriteInitialData())

        self._interface.initializeData()

        if self._interface.isReadDataAvailable():
            self._interface.readBlockScalarData(self._write_data_id, self._n_vertices, self._vertex_ids, self._write_data)

    def is_coupling_ongoing(self):
        if self._interface.isCouplingOngoing():
            if self._interface.isActionRequired(PySolverInterface.PyActionWriteIterationCheckpoint()):
                self._interface.fulfilledAction(PySolverInterface.PyActionWriteIterationCheckpoint())
            return True
        else:
            return False

    def check_convergence(self):
        if self._interface.isActionRequired(PySolverInterface.PyActionReadIterationCheckpoint()):
            self._interface.fulfilledAction(PySolverInterface.PyActionReadIterationCheckpoint())
            return False
        else:
            return True

    def extract_coupling_boundary_coordinates(self):
        vertices, _ = self.extract_coupling_boundary_vertices()
        vertices_x = vertices[0, :]
        vertices_y = vertices[1, :]
        if self._dimensions == 3:
            vertices_z = vertices[2, :]

        if self._dimensions == 2:
            return vertices_x, vertices_y
        elif self._dimensions == 3:
            # todo this has to be fixed for "proper" 3D coupling. Currently this is a workaround for the coupling of 2D fenics with pseudo 3D openfoam
            return vertices_x, vertices_y

    def finalize(self):
        self._interface.finalize()
