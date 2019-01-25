"""This module handles CustomExpression and initalization of the FEniCS
adapter.

:raise ImportError: if PRECICE_ROOT is not defined
"""
import dolfin
from dolfin import UserExpression, SubDomain, Function
from scipy.interpolate import Rbf
from scipy.interpolate import interp1d
import numpy as np
from .config import Config

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
        f = interp1d(self._coords_y, self._vals, bounds_error=False, fill_value="extrapolate")
        return f(x[1])

    def eval(self, value, x):
        value[0] = self.lin_interpol(x)


class Adapter:
    """Initializes the Adapter. Initalizer creates object of class Config (from
    config.py module).

    :ivar _config: object of class Config, which stores data from the JSON config file
    """
    def __init__(self, adapter_config_filename='precice-adapter-config.json'):

        self._config = Config(adapter_config_filename)

        self._solver_name = self._config.get_solver_name()

        self._interface = PySolverInterface.PySolverInterface(self._solver_name, 0, 1)
        self._interface.configure(self._config.get_config_file_name())
        self._dimensions = self._interface.getDimensions()

        self._coupling_subdomain = None # initialized later
        self._mesh_fenics = None # initialized later
        self._coupling_bc_expression = None # initialized later

        ## multirate time stepping
        self._N_this = self._config._N_this  # number of timesteps in this window, by default: no WR
        self._N_other = self._config._N_other  # number of timesteps in other window
        self._substep_counter = 0  # keeps track of number of substeps performed in window
        self._window_time = 0  # keeps track of window time

        ## coupling mesh related quantities
        self._coupling_mesh_vertices = None # initialized later
        self._mesh_name = self._config.get_coupling_mesh_name()
        self._mesh_id = self._interface.getMeshID(self._mesh_name)
        self._vertex_ids = None # initialized later
        self._n_vertices = None # initialized later

        ## write data related quantities (write data is written by this solver to preCICE)
        self._write_data_name = self._config.get_write_data_name()
        if self._waveform_relaxation_is_used():
            self._write_data_id = []
            for i in range(self._N_this + 1):
                self._write_data_id.append(self._interface.getDataID(self._write_data_name[i], self._mesh_id))
        else:
            self._write_data_id = self._interface.getDataID(self._write_data_name, self._mesh_id)
        self._write_data = None

        ## read data related quantities (read data is read by this solver from preCICE)
        self._read_data_name = self._config.get_read_data_name()
        if self._waveform_relaxation_is_used():
            self._read_data_id = []
            for i in range(self._N_other + 1):
                self._read_data_id.append(self._interface.getDataID(self._read_data_name[i], self._mesh_id))
        else:
            self._read_data_id = self._interface.getDataID(self._read_data_name, self._mesh_id)
        self._read_data = None

        ## numerics
        self._precice_tau = None

        ## checkpointing
        self._u_cp = None  # checkpoint for temperature inside domain
        self._t_cp = None  # time of the checkpoint
        self._n_cp = None  # timestep of the checkpoint

    def _window_size(self):
        return self._precice_tau

    def _reset_window_counters(self):
        self._substep_counter = 0
        self._window_time = 0

    def _waveform_relaxation_is_used(self):
        if self._N_this and self._N_this > 0 and self._N_other and self._N_other > 0:  # N_this and N_other are set, we want to use Waveform relaxation
            return True
        elif self._N_this:
            assert (self._N_this > 0)  # if key is defined, it has to be greater 0
        elif self._N_other:
            assert (self._N_other > 0)  # if key is defined, it has to be greater 0
        else:
            return False

    def convert_fenics_to_precice(self, data, mesh, subdomain):
        """Converts FEniCS data of type dolfin.Function into
        Numpy array for all x and y coordinates on the boundary.

        :param data: FEniCS boundary function
        :raise Exception: if type of data cannot be handled
        :return: array of FEniCS function values at each point on the boundary
        """
        if type(data) is dolfin.Function:
            x_all, y_all = self.extract_coupling_boundary_coordinates()
            return np.array([data(x, y) for x, y in zip(x_all, y_all)])
        else:
            raise Exception("Cannot handle data type %s" % type(data))

    def extract_coupling_boundary_vertices(self):
        """Extracts verticies which lay on the boundary. Currently handles 2D
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
                vertices_y.append(v.x(1))
                if self._dimensions == 3:
                    # todo this has to be fixed for "proper" 3D coupling. Currently this is a workaround for the coupling of 2D fenics with pseudo 3D openfoam
                    vertices_z.append(0)

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
        self._interface.setMeshVertices(self._mesh_id, self._n_vertices, self._coupling_mesh_vertices.flatten('F'), self._vertex_ids)

    def set_write_field(self, write_function_init):
        """Sets the write field. Called by initalize() function at the
        beginning of the simulation.

        :param write_function_init: function on the write field
        """
        if self._waveform_relaxation_is_used():
            for i in range(self._N_this + 1):
                self._write_data.append(self.convert_fenics_to_precice(write_function_init, self._mesh_fenics, self._coupling_subdomain))
            self._write_data[0] = None  # todo: nonsense code
            assert (self._write_data.__len__() == self._N_this + 1)
        else:
            self._write_data = self.convert_fenics_to_precice(write_function_init, self._mesh_fenics, self._coupling_subdomain)

    def set_read_field(self, read_function_init):
        """Sets the read field. Called by initalize() function at the
        beginning of the simulation.

        :param read_function_init: function on the read field
        """
        if self._waveform_relaxation_is_used():
            for i in range(self._N_other + 1):
                self._read_data.append(self.convert_fenics_to_precice(read_function_init, self._mesh_fenics, self._coupling_subdomain))
            self._read_data[0] = None  # todo: nonsense code
            assert (self._read_data.__len__() == self._N_other + 1)
        else:
            self._read_data = self.convert_fenics_to_precice(read_function_init, self._mesh_fenics, self._coupling_subdomain)

    def create_coupling_boundary_condition(self):
        """Creates the coupling boundary conditions using CustomExpression."""
        x_vert, y_vert = self.extract_coupling_boundary_coordinates()

        try:  # works with dolfin 1.6.0
            self._coupling_bc_expression = CustomExpression()
        except (TypeError, KeyError):  # works with dolfin 2017.2.0
            self._coupling_bc_expression = CustomExpression(degree=0)

        if self._waveform_relaxation_is_used():
            self._coupling_bc_expression.set_boundary_data(self._read_data[-1], x_vert, y_vert)
        else:
            self._coupling_bc_expression.set_boundary_data(self._read_data, x_vert, y_vert)

    def create_coupling_dirichlet_boundary_condition(self, function_space):
        """Creates the coupling Dirichlet boundary conditions using
        create_coupling_boundary_condition() method.

        :return: dolfin.DirichletBC()
        """
        self.create_coupling_boundary_condition()
        return dolfin.DirichletBC(function_space, self._coupling_bc_expression, self._coupling_subdomain)

    def create_coupling_neumann_boundary_condition(self, test_functions):
        """Creates the coupling Neumann boundary conditions using
        create_coupling_boundary_condition() method.

        :return: expression in form of integral: g*v*ds. (see e.g. p. 83ff
         Langtangen, Hans Petter, and Anders Logg. "Solving PDEs in Python The
         FEniCS Tutorial Volume I." (2016).)
        """
        self.create_coupling_boundary_condition()
        return self._coupling_bc_expression * test_functions * dolfin.ds  # this term has to be added to weak form to add a Neumann BC (see e.g. p. 83ff Langtangen, Hans Petter, and Anders Logg. "Solving PDEs in Python The FEniCS Tutorial Volume I." (2016).)

    def _window_is_completed(self):
        print("## window status:")
        print(self._window_size())
        print(self._window_time)
        print("##")
        if self._window_size() <= self._window_time:
            assert (self._substep_counter == self._N_this)
            assert (self._window_time == self._precice_tau)
            return True
        else:
            assert (self._substep_counter < self._N_this)
            return False

    def _do_interpolation(self, data, window_time):
        # this is currently a very limited dummy implementation

        # todo support "real" multirate, then remove following assertion
        assert(self._N_this == self._N_other)  # if self._N_this == self._N_other, we can assume that self._write_data = self._read_data and do not have to interpolate

        # todo support sampling data at arbitrary times
        assert(window_time * self._N_this % self._window_size() == 0)  # sampling time is exactly aligned with substep

        id_sample_at = round(window_time / self._window_size() * self._N_this)

        return data[id_sample_at]

    def _perform_substep(self, write_function, t, dt, n):
        x_vert, y_vert = self.extract_coupling_boundary_coordinates()
        # increase counters and window time
        self._window_time += dt
        self._substep_counter += 1
        assert(self._substep_counter > 0)
        assert(self._window_time / dt == self._substep_counter)  # we only support non-adaptive time stepping. Therefore i*dt == window time!
        assert(self._substep_counter <= self._N_this)

        # perform temporal interpolation on interface mesh
        interpolated_data = self._do_interpolation(self._read_data,
                                                   self._window_time)  # todo interpolate from other's time grid to this' time grid
        # store interface write data
        self._write_data[-1] = self.convert_fenics_to_precice(write_function, self._mesh_fenics, self._coupling_subdomain)  # todo HARDCODED!
        #self._write_data[self._substep_counter] = self.convert_fenics_to_precice(write_function, self._mesh_fenics, self._coupling_subdomain)  # todo should use this line

        # update interface read data
        self._coupling_bc_expression.update_boundary_data(self._read_data[-1], x_vert, y_vert)  # todo HARDCODED!
        #self._coupling_bc_expression.update_boundary_data(interpolated_data, x_vert, y_vert)  # todo should use this line

        t += dt
        n += 1
        success = True
        
        return t, n, success

    def advance(self, write_function, u_np1, u_n, t, dt, n):
        """Calls preCICE advance function using PySolverInterface and manages checkpointing.
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

        if self._waveform_relaxation_is_used():
            t, n, success = self._perform_substep(write_function, t, dt, n)

        if not self._waveform_relaxation_is_used() or self._window_is_completed():  # window completed
            precice_step_complete = False

            # communication
            if self._waveform_relaxation_is_used():
                for i in range(1, self._N_this + 1):  # todo should start at 0
                    self._interface.writeBlockScalarData(self._write_data_id[i], self._n_vertices, self._vertex_ids, self._write_data[i])
            else:
                x_vert, y_vert = self.extract_coupling_boundary_coordinates()
                self._write_data = self.convert_fenics_to_precice(write_function, self._mesh_fenics, self._coupling_subdomain)
                self._interface.writeBlockScalarData(self._write_data_id, self._n_vertices, self._vertex_ids, self._write_data)
            max_dt = self._interface.advance(dt)
            if self._waveform_relaxation_is_used():
                for i in range(1, self._N_other + 1):  # todo should start at 0
                    self._interface.readBlockScalarData(self._read_data_id[i], self._n_vertices, self._vertex_ids, self._read_data[i])
                assert(False)  # TODO possible reason for bug here? Should we update the coupling bc at this point?
            else:
                self._interface.readBlockScalarData(self._read_data_id, self._n_vertices, self._vertex_ids, self._read_data)
                self._coupling_bc_expression.update_boundary_data(self._read_data, x_vert, y_vert)  # TODO: this should go somewhere inside _perform_substep, however, if we do not use Waveform relaxation, we have to run the command after calling advance and readBlockScalarData

            # checkpointing
            if self._interface.isActionRequired(PySolverInterface.PyActionReadIterationCheckpoint()):
                # continue FEniCS computation from checkpoint
                u_n.assign(self._u_cp)  # set u_n to value of checkpoint
                t = self._t_cp
                n = self._n_cp
                self._interface.fulfilledAction(PySolverInterface.PyActionReadIterationCheckpoint())
            else:
                u_n.assign(u_np1)
                t = new_t = t + dt  # todo the variables new_t, new_n could be saved, by just using t and n below, however I think it improved readability.
                n = new_n = n + 1

            if self._interface.isActionRequired(PySolverInterface.PyActionWriteIterationCheckpoint()):
                # continue FEniCS computation with u_np1
                # update checkpoint
                self._u_cp.assign(u_np1)
                self._t_cp = new_t
                self._n_cp = new_n
                if self._waveform_relaxation_is_used():
                    # todo the following part of code is really cryptic. we should really refactor it
                    """
                    What's the actual purpose: We never update self._write_data[0], but we have to keep it for 
                    interpolation. Therefore, after an iteration has ended successfully, we have to use the last sample of 
                    the successful iteration as the first sample for the next. Additionally, we need an initial guess for
                    the read data.
                    """
                    initial_guess_write = np.copy(self._write_data[-1])
                    for i in range(self._N_this + 1):
                        self._write_data[i] = np.copy(initial_guess_write)
                    initial_guess_read = np.copy(self._read_data[-1])
                    for i in range(self._N_other + 1):
                        self._read_data[i] = np.copy(initial_guess_read)
                    """
                    until here
                    """
                self._interface.fulfilledAction(PySolverInterface.PyActionWriteIterationCheckpoint())
                precice_step_complete = True

            if self._waveform_relaxation_is_used():
                self._reset_window_counters()

        return t, n, precice_step_complete, max_dt

    def initialize(self, coupling_subdomain, mesh, read_field, write_field, u_n, t=0, n=0):
        """Initializes remaining attributes. Called once, from the solver.

        :param read_field: function applied on the read field
        :param write_field: function applied on the write field
        """
        self.set_coupling_mesh(mesh, coupling_subdomain)
        self.set_read_field(read_field)
        self.set_write_field(write_field)
        self._precice_tau = self._interface.initialize()

        if self._interface.isActionRequired(PySolverInterface.PyActionWriteInitialData()):
            if self._waveform_relaxation_is_used():
                for i in range(1, self._N_this + 1):  # todo should start at 0
                    self._interface.writeBlockScalarData(self._write_data_id[i], self._n_vertices, self._vertex_ids, self._write_data[i])
            else:
                self._interface.writeBlockScalarData(self._write_data_id, self._n_vertices, self._vertex_ids, self._write_data)
            self._interface.fulfilledAction(PySolverInterface.PyActionWriteInitialData())

        self._interface.initializeData()

        if self._interface.isReadDataAvailable():
            if self._waveform_relaxation_is_used():
                for i in range(1, self._N_other + 1):  # todo should start at 0
                    self._interface.readBlockScalarData(self._read_data_id[i], self._n_vertices, self._vertex_ids, self._read_data[i])
            else:
                self._interface.readBlockScalarData(self._read_data_id, self._n_vertices, self._vertex_ids, self._read_data)

        if self._interface.isActionRequired(PySolverInterface.PyActionWriteIterationCheckpoint()):
            self._u_cp = u_n.copy(deepcopy=True)
            self._t_cp = t
            self._n_cp = n
            self._interface.fulfilledAction(PySolverInterface.PyActionWriteIterationCheckpoint())

        return self._precice_tau

    def is_coupling_ongoing(self):
        """Determines whether simulation should continue. Called from the
        simulation loop in the solver.

        :return: True if the coupling is ongoing, False otherwise
        """
        return self._interface.isCouplingOngoing()

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
        elif self._dimensions == 3:
            # todo this has to be fixed for "proper" 3D coupling. Currently this is a workaround for the coupling of 2D fenics with pseudo 3D openfoam
            return vertices_x, vertices_y

    def finalize(self):
        """Finalizes the coupling interface."""
        self._interface.finalize()
