import numpy as np

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
    def configure_waveform_relaxation(self, adapter_config_filename='precice-adapter-config-WR.json'):
        self._sample_counter_this = 0
        self._sample_counter_other = 0

        self._config = Config(adapter_config_filename)
        self._precice_tau = None

        # multirate time stepping
        self._N_this = self._config._N_this  # number of timesteps in this window, by default: no WR
        self._N_other = self._config._N_other  # number of timesteps in other window
        self._current_window_start = 0  # defines start of window
        self._window_time = self._current_window_start  # keeps track of window time

    def initialize_waveforms(self, mesh_id , n_vertices, vertex_ids, write_data_name, read_data_name, n_data):
        # constant information of mesh
        self._mesh_id = mesh_id
        self._n_vertices = n_vertices
        self._vertex_ids = vertex_ids

        # constant write data name prefix
        self._write_data_name = write_data_name
        self._write_data_buffer = np.zeros(n_data)  # TODO later, we want to have more than one sample in this buffer to be able to interpolate

        # constant read data name prefix
        self._read_data_name = read_data_name
        self._read_data_buffer = np.zeros(n_data)  # TODO later, we want to have more than one sample in this buffer to be able to interpolate

    def writeBlockScalarData(self, write_data_name, mesh_id, n_vertices, vertex_ids, write_data, time):
        assert(self._config.get_write_data_name() == write_data_name)
        assert(self._is_inside_current_window(time))
        # we put the data into a buffer. Data will be send to other participant via preCICE in advance
        self._write_data_buffer = write_data[:]
        # we assert that the preCICE specific write parameters did not change since configure_waveform_relaxation
        assert (self._mesh_id == mesh_id)
        assert (self._n_vertices == n_vertices)
        assert (self._vertex_ids == vertex_ids)
        assert (self._write_data_name == write_data_name)

    def readBlockScalarData(self, read_data_name, mesh_id, n_vertices, vertex_ids, read_data, time):
        assert(self._config.get_read_data_name() == read_data_name)
        assert(self._is_inside_current_window(time))
        # we get the data from the interpolant. New data will be obtained from the other participant via preCICE in advance
        read_data[:] = self._read_data_buffer
        # we assert that the preCICE specific write parameters did not change since configure_waveform_relaxation
        assert (self._mesh_id == mesh_id)
        assert (self._n_vertices == n_vertices)
        assert (self._vertex_ids == vertex_ids)
        assert (self._read_data_name == read_data_name)

    def advance(self, dt):
        self._window_time += dt
        if self._window_is_completed():
            print("WINDOW COMPLETE!")
            write_data_id = self.getDataID(self._write_data_name, self._mesh_id)
            read_data_id = self.getDataID(self._read_data_name, self._mesh_id)
            super().writeBlockScalarData(write_data_id, self._n_vertices, self._vertex_ids, self._write_data_buffer)
            max_dt = super().advance(self._window_time)  # = time given by preCICE
            super().readBlockScalarData(read_data_id, self._n_vertices, self._vertex_ids, self._read_data_buffer)

            # checkpointing
            if self.isActionRequired(PySolverInterface.PyActionReadIterationCheckpoint()):
                print("REPEAT WINDOW!")
                # repeat window
            else:
                print("GO TO NEXT WINDOW!")
                # go to next window
                self._current_window_start += self._window_time

            self._reset_window()
        else:
            print("WINDOW INCOMPLETE!")
            max_dt = self._remaining_window_time()  # = window time remaining
            assert(max_dt > 0)

        return max_dt

    def _print_window_status(self):
        print("## window status:")
        print(self._current_window_start)
        print(self._window_size())
        print(self._window_time)
        print("##")

    def _window_is_completed(self):
        if self._window_size() <= self._window_time:
            assert (self._window_time == self._precice_tau)
            return True
        else:
            return False

    def _remaining_window_time(self):
        return self._window_size() - self._window_time

    def _current_window_end(self):
        return self._current_window_start + self._window_size()

    def _is_inside_current_window(self, global_time):
        local_time = global_time - self._current_window_start
        tol = self._window_size() * 10**-5
        return 0-tol <= local_time <= self._window_size()+tol

    def _window_size(self):
        return self._precice_tau

    def _reset_window(self):
        self._window_time = 0
        self._write_data_buffer = dict()
        self._read_data_buffer = dict()

    def _perform_substep(self, write_function, t, dt, n):
        # increase counters and window time
        self._window_time += dt

        # perform temporal interpolation on interface mesh
        # TODO
        # store interface write data
        # TODO
        # update interface read data
        # TODO

        t += dt
        n += 1
        success = True

        return t, n, success

    def initialize(self):
        self._precice_tau = super().initialize()
        return np.max([self._precice_tau, self._remaining_window_time()])

    def initializeData(self):
        return super().initializeData()

    def _do_interpolation(self, data, window_time):
        # this is currently a very limited dummy implementation

        # todo support "real" multirate, then remove following assertion
        assert(self._N_this == self._N_other)  # if self._N_this == self._N_other, we can assume that self._write_data = self._read_data and do not have to interpolate

        # todo support sampling data at arbitrary times
        assert(window_time * self._N_this % self._window_size() == 0)  # sampling time is exactly aligned with substep

        id_sample_at = round(window_time / self._window_size() * self._N_this)

        return data[id_sample_at]
