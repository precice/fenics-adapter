try:
    import PySolverInterface
    from PySolverInterface import PyActionReadIterationCheckpoint, PyActionWriteInitialData, PyActionWriteIterationCheckpoint
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

from .config import Config


class WaveformBindings(PySolverInterface.PySolverInterface):
    def __init__(self, name, rank, procs, adapter_config_filename='precice-adapter-config-WR.json'):
        print("INIT CALLED!")

        self._sample_counter_this = 0
        self._sample_counter_other = 0

        self._config = Config(adapter_config_filename)

        ## multirate time stepping
        self._N_this = self._config._N_this  # number of timesteps in this window, by default: no WR
        self._N_other = self._config._N_other  # number of timesteps in other window
        self._substep_counter = 0  # keeps track of number of substeps performed in window
        self._window_time = 0  # keeps track of window time

        super().__init__()

    def __new__(cls, name, rank, procs):
        print("NEW CALLED!")
        return super().__new__(cls, name, rank, procs)

    def writeBlockScalarData(self, write_data_name, mesh_id, n_vertices, vertex_ids, write_data):
        assert(self._config.get_write_data_name() == write_data_name)
        self._sample_counter_this += 1
        # write_data_name = write_data_name+self._sample_counter_this
        write_data_id = super().getDataID(write_data_name, mesh_id)
        super().writeBlockScalarData(write_data_id, n_vertices, vertex_ids, write_data)
        # TODO here, we either access preCICE or put the data into a buffer

    def readBlockScalarData(self, read_data_name, mesh_id, n_vertices, vertex_ids, read_data):
        assert(self._config.get_read_data_name() == read_data_name)
        self._sample_counter_other += 1
        # read_data_name = read_data_name+self._sample_counter_other
        read_data_id = super().getDataID(read_data_name, mesh_id)
        super().readBlockScalarData(read_data_id, n_vertices, vertex_ids, read_data)
        # TODO here, we either access preCICE or put the data into a buffer

    def advance(self):
        # TODO here, we either call preCICE.advance() or update FEniCS boundary conditions from the WR interpolation
        return super().advance()

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

    def _window_size(self):  # TODO: Remove this block and put it into WR Bindings
        return self._precice_tau

    def _reset_window_counters(self):  # TODO: Remove this block and put it into WR Bindings
        self._substep_counter = 0
        self._window_time = 0

    def _waveform_relaxation_is_used(self):  # TODO: Remove this block and put it into WR Bindings
        if self._N_this and self._N_this > 0 and self._N_other and self._N_other > 0:  # N_this and N_other are set, we want to use Waveform relaxation
            return True
        elif self._N_this:
            assert (self._N_this > 0)  # if key is defined, it has to be greater 0
        elif self._N_other:
            assert (self._N_other > 0)  # if key is defined, it has to be greater 0
        else:
            return False

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

    def initialize(self):
        return super().initialize()

    def initializeData(self):
        return super().initializeData()
