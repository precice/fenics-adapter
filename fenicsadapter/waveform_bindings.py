import numpy as np

try:
    import precice
    from precice import action_read_iteration_checkpoint, action_write_initial_data, action_write_iteration_checkpoint
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
    from precice import action_read_iteration_checkpoint, action_write_initial_data, action_write_iteration_checkpoint

from .config import Config

class WaveformBindings(precice.Interface):
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

        # number of samples in one window
        self._n_data = n_data

        # constant write data name prefix
        self._write_data_name = write_data_name
        self._write_data_buffer = self._get_empty_buffer()  # TODO later, we want to have more than one sample in this buffer to be able to interpolate

        # constant read data name prefix
        self._read_data_name = read_data_name
        self._read_data_buffer = self._get_empty_buffer()  # TODO later, we want to have more than one sample in this buffer to be able to interpolate

    def _get_empty_buffer(self):
        buffer=[]
        for _ in range(self._n_data):
            buffer.append(np.zeros(self._n_vertices))
        return buffer

    def write_block_scalar_data(self, write_data_name, mesh_id, n_vertices, vertex_ids, write_data, time):
        assert(self._config.get_write_data_name() == write_data_name)
        assert(self._is_inside_current_window(time))
        # we put the data into a buffer. Data will be send to other participant via preCICE in advance
        self._write_data_buffer[0] = write_data[:]  # todo currently, our buffer only stores one sample. Later, we want to write the data depending on the given "time"
        # we assert that the preCICE specific write parameters did not change since configure_waveform_relaxation
        assert (self._mesh_id == mesh_id)
        assert (self._n_vertices == n_vertices)
        assert (self._vertex_ids == vertex_ids)
        assert (self._write_data_name == write_data_name)

    def read_block_scalar_data(self, read_data_name, mesh_id, n_vertices, vertex_ids, read_data, time):
        assert(self._config.get_read_data_name() == read_data_name)
        assert(self._is_inside_current_window(time))
        # we get the data from the interpolant. New data will be obtained from the other participant via preCICE in advance
        read_data[:] = self._read_data_buffer[0]  # todo currently, our buffer only stores one sample. Later, we want to read the data depending on the given "time"
        # we assert that the preCICE specific write parameters did not change since configure_waveform_relaxation
        assert (self._mesh_id == mesh_id)
        assert (self._n_vertices == n_vertices)
        assert (self._vertex_ids == vertex_ids)
        assert (self._read_data_name == read_data_name)

    def advance(self, dt):
        self._window_time += dt
        if self._window_is_completed():
            print("WINDOW COMPLETE!")
            write_data_id = self.get_data_id(self._write_data_name, self._mesh_id)
            read_data_id = self.get_data_id(self._read_data_name, self._mesh_id)
            super().write_block_scalar_data(write_data_id, self._n_vertices, self._vertex_ids, self._write_data_buffer)
            max_dt = super().advance(self._window_time)  # = time given by preCICE
            super().write_block_scalar_data(read_data_id, self._n_vertices, self._vertex_ids, self._read_data_buffer)

            # checkpointing
            if self.is_action_required(action_read_iteration_checkpoint()):
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
        self._write_data_buffer = self._get_empty_buffer()
        self._read_data_buffer = self._get_empty_buffer()

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

    def initialize_data(self):
        return super().initialize_data()

    def _do_interpolation(self, data, window_time):
        # this is currently a very limited dummy implementation

        # todo support "real" multirate, then remove following assertion
        assert(self._N_this == self._N_other)  # if self._N_this == self._N_other, we can assume that self._write_data = self._read_data and do not have to interpolate

        # todo support sampling data at arbitrary times
        assert(window_time * self._N_this % self._window_size() == 0)  # sampling time is exactly aligned with substep

        id_sample_at = round(window_time / self._window_size() * self._N_this)

        return data[id_sample_at]


class OutOfWindowError(Exception):
    """Raised when the time is not inside the window"""
    pass

class NotInTemporalGridError(Exception):
    """Raised when the point in time is not in the temporal grid"""
    pass


class Waveform:
    def __init__(self, temporal_grid, window_start, window_size):
        """
        :param temporal_grid: temporal grid in local coordinates in [0,1]
        :param window_start: starting time of the window
        :param window_size: size of window
        """
        assert (abs(temporal_grid[0] - 0) < 10**-10)
        assert (abs(temporal_grid[-1] - 1) < 10**-10)
        assert (window_size > 0)
        self._temporal_grid = temporal_grid
        self._samples_in_time = dict()
        self._window_size = window_size
        self._window_start = window_start
        for t in self._temporal_grid:
            self._samples_in_time[t] = None

    def initialize(self, data):
        self._n_datapoints = data.shape[0]
        for t in self._temporal_grid:
            self._samples_in_time[t] = data.copy()

    def _sample(self, local_time):
        from scipy.interpolate import interp1d

        if not (0 <= local_time <= 1):
            raise OutOfWindowError

        return_value = np.zeros(self._n_datapoints)
        for i in range(self._n_datapoints):
            values_along_time = dict()
            for t in self._temporal_grid:
                values_along_time[t] = self._samples_in_time[t][i]
            interpolant = interp1d(list(values_along_time.keys()), list(values_along_time.values()))
            return_value[i] = interpolant(local_time)
        return return_value

    def sample(self, global_time):
        local_time = self.global_to_local_time(global_time)
        return self._sample(local_time)

    def global_to_local_time(self, global_time):
        return (global_time - self._window_start)/self._window_size

    def global_temporal_grid(self):
        return self._temporal_grid * self._window_size + self._window_start

    def update(self, data, global_time):
        if not (global_time in self.global_temporal_grid()):
            raise NotInTemporalGridError
        assert (data.shape[0] == self._n_datapoints)
        self._samples_in_time[self.global_to_local_time(global_time)] = data
