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

        ## multirate time stepping
        self._N_this = self._config._N_this  # number of timesteps in this window, by default: no WR
        self._N_other = self._config._N_other  # number of timesteps in other window
        self._current_window_start = 0  # defines start of window
        self._window_time = self._current_window_start  # keeps track of window time
        self._write_data_buffer = dict()
        self._read_data_buffer = dict()

    def writeBlockScalarData(self, write_data_name, mesh_id, n_vertices, vertex_ids, write_data, time):
        assert(self._config.get_write_data_name() == write_data_name)
        assert(self._is_inside_current_window(time))
        self._write_data_buffer[time] = write_data
        # write_data_name = write_data_name+self._sample_counter_this
        write_data_id = self.getDataID(write_data_name, mesh_id)
        # TODO here, we either access preCICE or put the data into a buffer
        super().writeBlockScalarData(write_data_id, n_vertices, vertex_ids, write_data)

    def readBlockScalarData(self, read_data_name, mesh_id, n_vertices, vertex_ids, read_data, time):
        assert(self._config.get_read_data_name() == read_data_name)
        assert(self._is_inside_current_window(time))
        # read_data_name = read_data_name+self._sample_counter_other
        read_data_id = self.getDataID(read_data_name, mesh_id)
        # TODO here, we either access preCICE or put the data into a buffer
        super().readBlockScalarData(read_data_id, n_vertices, vertex_ids, read_data)
        self._read_data_buffer[time] = read_data

    def advance(self, dt):
        self._window_time += dt
        if self._window_is_completed():
            print("WINDOW COMPLETE!")

            max_dt = super().advance(self._window_time)  # = time given by preCICE

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
