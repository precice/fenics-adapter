import dolfin
from dolfin import Expression, SubDomain
from scipy.interpolate import Rbf
from scipy.interpolate import interp1d
import numpy as np

try:
    import PySolverInterface
except ImportError:
    import os
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
        self.coords_x = coords_x
        if coords_y is None:
            coords_y = np.zeros(coords_x.shape)
        self.coords_y = coords_y
        if coords_z is None:
            coords_z = np.zeros(coords_x.shape)
        self.coords_z = coords_z

        self.vals = vals.flatten()
        assert (self.vals.shape == self.coords_x.shape)

    def rbf_interpol(self, x):
        if x.__len__() == 1:
            f = Rbf(self.coords_x, self.vals.flatten())
            return f(x)
        if x.__len__() == 2:
            f = Rbf(self.coords_x, self.coords_y, self.vals.flatten())
            return f(x[0], x[1])
        if x.__len__() == 3:
            f = Rbf(self.coords_x, self.coords_y, self.coords_z, self.vals.flatten())
            return f(x[0], x[1], x[2])

    def lin_interpol(self, x):
        f = interp1d(self.coords_x, self.vals)
        return f(x.x())

    def eval(self, value, x):
        value[0] = self.rbf_interpol(x)


class Coupling(object):
    def __init__(self, config_file_name, solver_name):
        self.solverName = solver_name  # name of the solver, like defined in the config

        self.interface = PySolverInterface.PySolverInterface(solver_name, 0, 1)
        self.interface.configure(config_file_name)
        self.dimensions = self.interface.getDimensions()

        self.coupling_subdomain = None  # FEniCS subdomain defining the coupling interface
        self.mesh_fenics = None  # FEniCS mesh where the coupled simulation takes place
        self.coupling_bc_expression = None  # Expression describing the coupling condition

        ## coupling mesh related quantities will be defined later
        self.coupling_mesh_vertices = None  # mesh vertices in a format that can be understood by preCICE
        self.mesh_id = None  # ID of the coupling mesh created from mesh name
        self.vertex_ids = None  # ID of vertices, will be filled by preCICE
        self.n_vertices = None  # number of vertices

        ## write data related quantities will be defined later (write data is written by this solver to preCICE)
        self.write_data_id = None  # ID of the data on the coupling mesh created from data name
        self.write_data = None  # actual data

        ## read data related quantities will be defined later (read data is read by this solver from preCICE)
        self.read_data_id = None  # ID of the data on the coupling mesh created from data name
        self.read_data = None  # actual data

        ## numerics
        self.precice_tau = None

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
        if self.dimensions == 3:
            vertices_z = []

        if not issubclass(type(self.coupling_subdomain), SubDomain):
            raise Exception("no correct coupling interface defined!")

        for v in dolfin.vertices(self.mesh_fenics):
            if self.coupling_subdomain.inside(v.point(), True):
                n += 1
                vertices_x.append(v.x(0))
                vertices_y.append(v.x(1))
                if self.dimensions == 3:
                    # todo this has to be fixed for "proper" 3D coupling. Currently this is a workaround for the coupling of 2D fenics with pseudo 3D openfoam
                    vertices_z.append(0)

        if self.dimensions == 2:
            return np.stack([vertices_x, vertices_y]), n
        elif self.dimensions == 3:
            return np.stack([vertices_x, vertices_y, vertices_z]), n

    def set_coupling_mesh(self, mesh, subdomain, coupling_mesh_name):
        self.coupling_subdomain = subdomain
        self.mesh_fenics = mesh
        self.coupling_mesh_vertices, self.n_vertices = self.extract_coupling_boundary_vertices()
        self.mesh_id = self.interface.getMeshID(coupling_mesh_name)
        self.vertex_ids = np.zeros(self.n_vertices)
        self.interface.setMeshVertices(self.mesh_id, self.n_vertices, self.coupling_mesh_vertices.flatten('F'), self.vertex_ids)

    def set_write_field(self, write_data_name, write_function_init):
        self.write_data_id = self.interface.getDataID(write_data_name, self.mesh_id)
        self.write_data = self.convert_fenics_to_precice(write_function_init, self.mesh_fenics, self.coupling_subdomain)

    def set_read_field(self, read_data_name, read_function_init):
        self.read_data_id = self.interface.getDataID(read_data_name, self.mesh_id)
        self.read_data = self.convert_fenics_to_precice(read_function_init, self.mesh_fenics, self.coupling_subdomain)

    def create_coupling_boundary_condition(self):
        x_vert, y_vert = self.extract_coupling_boundary_coordinates()

        try:  # works with dolfin 1.6.0
            self.coupling_bc_expression = CustomExpression(self.read_data, x_vert, y_vert)
        except (TypeError, KeyError):  # works with dolfin 2017.2.0
            self.coupling_bc_expression = CustomExpression(self.read_data, x_vert, y_vert, degree=0)

    def create_coupling_dirichlet_boundary_condition(self, function_space):
        self.create_coupling_boundary_condition()
        return dolfin.DirichletBC(function_space, self.coupling_bc_expression, self.coupling_subdomain)

    def create_coupling_neumann_boundary_condition(self, test_functions):
        self.create_coupling_boundary_condition()
        return self.coupling_bc_expression * test_functions * dolfin.ds  # this term has to be added to weak form to add a Neumann BC (see e.g. p. 83ff Langtangen, Hans Petter, and Anders Logg. "Solving PDEs in Python The FEniCS Tutorial Volume I." (2016).)

    def exchange_data(self, write_function, dt):
        x_vert, y_vert = self.extract_coupling_boundary_coordinates()
        self.write_data = self.convert_fenics_to_precice(write_function, self.mesh_fenics, self.coupling_subdomain)
        self.interface.writeBlockScalarData(self.write_data_id, self.n_vertices, self.vertex_ids, self.write_data)
        self.interface.advance(dt)
        self.interface.readBlockScalarData(self.read_data_id, self.n_vertices, self.vertex_ids, self.read_data)
        self.coupling_bc_expression.update_boundary_data(self.read_data, x_vert, y_vert)

    def initialize_data(self):
        self.precice_tau = self.interface.initialize()

        if self.interface.isActionRequired(PySolverInterface.PyActionWriteInitialData()):
            self.interface.writeBlockScalarData(self.read_data_id, self.n_vertices, self.vertex_ids, self.read_data)
            self.interface.fulfilledAction(PySolverInterface.PyActionWriteInitialData())

        self.interface.initializeData()

        if self.interface.isReadDataAvailable():
            self.interface.readBlockScalarData(self.write_data_id, self.n_vertices, self.vertex_ids, self.write_data)

    def is_coupling_ongoing(self):
        if self.interface.isCouplingOngoing():
            if self.interface.isActionRequired(PySolverInterface.PyActionWriteIterationCheckpoint()):
                self.interface.fulfilledAction(PySolverInterface.PyActionWriteIterationCheckpoint())
            return True
        else:
            return False

    def check_convergence(self):
        if self.interface.isActionRequired(PySolverInterface.PyActionReadIterationCheckpoint()):
            self.interface.fulfilledAction(PySolverInterface.PyActionReadIterationCheckpoint())
            return False
        else:
            return True

    def extract_coupling_boundary_coordinates(self):
        vertices, _ = self.extract_coupling_boundary_vertices()
        vertices_x = vertices[0, :]
        vertices_y = vertices[1, :]
        if self.dimensions == 3:
            vertices_z = vertices[2, :]

        if self.dimensions == 2:
            return vertices_x, vertices_y
        elif self.dimensions == 3:
            # todo this has to be fixed for "proper" 3D coupling. Currently this is a workaround for the coupling of 2D fenics with pseudo 3D openfoam
            return vertices_x, vertices_y

    def finalize(self):
        self.interface.finalize()
