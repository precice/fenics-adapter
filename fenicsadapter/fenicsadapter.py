"""This module handles high-level functionality of the FEniCS adapter.
"""
import dolfin
import fenicsadapter.core
from .config import Config


class Adapter:
    """Initializes the Adapter. Initalizer creates object of class Config (from
    config.py module).

    :ivar _config: object of class Config, which stores data from the JSON config file
    """
    def __init__(self, adapter_config_filename='precice-adapter-config.json'):

        self._config = Config(adapter_config_filename)

        self._coupling_bc_expression = None  # initialized later

        self.adapter = fenicsadapter.core.Adapter(self._config.get_solver_name(), 0, 1)
        self.adapter.configure(self._config.get_config_file_name())


        ## identifies mesh, write and read data
        self._mesh_name = self._config.get_coupling_mesh_name()
        self._write_data_name = self._config.get_write_data_name()
        self._read_data_name = self._config.get_read_data_name()

        ## numerics
        self._precice_tau = None

        ## checkpointing
        self._u_cp = None  # checkpoint for temperature inside domain
        self._t_cp = None  # time of the checkpoint
        self._n_cp = None  # timestep of the checkpoint

    def create_coupling_dirichlet_boundary_condition(self, function_space):
        """Creates the coupling Dirichlet boundary conditions using
        create_coupling_boundary_condition() method.

        :return: dolfin.DirichletBC()
        """
        return dolfin.DirichletBC(function_space, self._coupling_bc_expression, self.adapter._coupling_subdomain)

    def create_coupling_neumann_boundary_condition(self, test_functions):
        """Creates the coupling Neumann boundary conditions using
        create_coupling_boundary_condition() method.

        :return: expression in form of integral: g*v*ds. (see e.g. p. 83ff
         Langtangen, Hans Petter, and Anders Logg. "Solving PDEs in Python The
         FEniCS Tutorial Volume I." (2016).)
        """
        return self._coupling_bc_expression * test_functions * dolfin.ds  # this term has to be added to weak form to add a Neumann BC (see e.g. p. 83ff Langtangen, Hans Petter, and Anders Logg. "Solving PDEs in Python The FEniCS Tutorial Volume I." (2016).)

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

        self.adapter.write_block_scalar_data(self._write_data_name, self._mesh_name, write_function)
        max_dt = self.adapter.advance(dt)
        read_expression = self.adapter.read_block_scalar_data(self._read_data_name, self._mesh_name)
        self._coupling_bc_expression.update(read_expression)

        precice_step_complete = False

        # checkpointing
        if self.adapter.is_action_required(fenicsadapter.core.action_read_iteration_checkpoint()):
            # continue FEniCS computation from checkpoint
            u_n.assign(self._u_cp)  # set u_n to value of checkpoint
            t = self._t_cp
            n = self._n_cp
            self.adapter.fulfilled_action(fenicsadapter.core.action_read_iteration_checkpoint())
        else:
            u_n.assign(u_np1)
            t = new_t = t + dt  # todo the variables new_t, new_n could be saved, by just using t and n below, however I think it improved readability.
            n = new_n = n + 1

        if self.adapter.is_action_required(fenicsadapter.core.action_write_iteration_checkpoint()):
            # continue FEniCS computation with u_np1
            # update checkpoint
            self._u_cp.assign(u_np1)
            self._t_cp = new_t
            self._n_cp = new_n
            self.adapter.fulfilled_action(fenicsadapter.core.action_write_iteration_checkpoint())
            precice_step_complete = True

        return t, n, precice_step_complete, max_dt

    def initialize(self, coupling_subdomain, mesh, read_field, write_field, u_n, t=0, n=0):
        """Initializes remaining attributes. Called once, from the solver.

        :param read_field: function applied on the read field
        :param write_field: function applied on the write field
        """
        self.adapter.set_coupling_mesh(self._mesh_name, mesh, coupling_subdomain)
        self._precice_tau = self.adapter.initialize()

        if self.adapter.is_action_required(fenicsadapter.core.action_write_initial_data()):
            self.adapter.write_block_scalar_data(self._write_data_name, self._mesh_name, write_field)
            self.adapter.fulfilled_action(fenicsadapter.core.action_write_initial_data())

        self.adapter.initialize_data()

        if self.adapter.is_read_data_available():
            read_expression = self.adapter.read_block_scalar_data(self._read_data_name, self._mesh_name)
        else:
            read_expression = self.adapter.create_coupling_boundary_condition(read_field)
        self._coupling_bc_expression = read_expression

        if self.adapter.is_action_required(fenicsadapter.core.action_write_iteration_checkpoint()):
            self._u_cp = u_n.copy(deepcopy=True)
            self._t_cp = t
            self._n_cp = n
            self.adapter.fulfilled_action(fenicsadapter.core.action_write_iteration_checkpoint())

        return self._precice_tau

    def is_coupling_ongoing(self):
        """Determines whether simulation should continue. Called from the
        simulation loop in the solver.

        :return: True if the coupling is ongoing, False otherwise
        """
        return self.adapter.is_coupling_ongoing()

    def finalize(self):
        """Finalizes the coupling interface."""
        self.adapter.finalize()

    def get_solver_name(self):
        return self._config.get_solver_name()
