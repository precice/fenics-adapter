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

    def configure_waveform_relaxation(self, adapter_config_filename, other_adapter_config_filename):
        self._sample_counter_this = 0
        self._sample_counter_other = 0

        self._config = Config(adapter_config_filename)
        self._other_config = Config(other_adapter_config_filename)
        self._precice_tau = None

        # multirate time stepping
        self._n_this = self._config.get_n_substeps() + 1  # number of timesteps in this window, by default: no WR
        self._n_other = self._other_config.get_n_substeps() + 1 # number of timesteps in other window, todo: in the end we don't want to worry about the other solver's resolution!
        self._current_window_start = 0  # defines start of window
        self._window_time = self._current_window_start  # keeps track of window time

    def initialize_waveforms(self, mesh_id, n_vertices, vertex_ids, write_data_name, read_data_name,
                             write_data_init, read_data_init):
        print("INIT WAVEFORMS!")
        # constant information of mesh
        self._mesh_id = mesh_id
        self._n_vertices = n_vertices
        self._vertex_ids = vertex_ids

        # constant write data name prefix
        self._write_data_name = write_data_name
        self._write_data_buffer = self._get_empty_write_buffer(write_data_init)

        # constant read data name prefix
        self._read_data_name = read_data_name
        self._read_data_buffer = self._get_empty_read_buffer(read_data_init)

    def _get_empty_write_buffer(self, write_data_init):
        buffer = Waveform(self._current_window_start, self._precice_tau, self._n_vertices)
        buffer.append(write_data_init, self._current_window_start)
        return buffer

    def _get_empty_read_buffer(self, read_data_init):
        buffer = Waveform(self._current_window_start, self._precice_tau, self._n_vertices)
        buffer.initialize_constant(read_data_init)
        return buffer

    def write_block_scalar_data(self, write_data_name, mesh_id, n_vertices, vertex_ids, write_data, time):
        assert(self._config.get_write_data_name() == write_data_name)
        assert(self._is_inside_current_window(time))
        # we put the data into a buffer. Data will be send to other participant via preCICE in advance
        self._write_data_buffer.append(write_data[:], time)
        # we assert that the preCICE specific write parameters did not change since configure_waveform_relaxation
        assert (self._mesh_id == mesh_id)
        assert (self._n_vertices == n_vertices)
        assert ((self._vertex_ids == vertex_ids).all())
        assert (self._write_data_name == write_data_name)

    def read_block_scalar_data(self, read_data_name, mesh_id, n_vertices, vertex_ids, read_data, time):
        assert(self._config.get_read_data_name() == read_data_name)
        assert(self._is_inside_current_window(time))
        # we get the data from the interpolant. New data will be obtained from the other participant via preCICE in advance
        print("read at time {time}".format(time=time))
        read_data[:] = self._read_data_buffer.sample(time)[:].copy()
        print("read_data is {read_data}".format(read_data=read_data))
        # we assert that the preCICE specific write parameters did not change since configure_waveform_relaxation
        assert (self._mesh_id == mesh_id)
        assert (self._n_vertices == n_vertices)
        assert ((self._vertex_ids == vertex_ids).all())
        assert (self._read_data_name == read_data_name)

    def _write_all_window_data_to_precice(self):
        write_data_name_prefix = self._write_data_name
        write_waveform = self._write_data_buffer
        for substep in range(self._n_this):
            write_data_name = write_data_name_prefix + str(substep)
            write_data_id = self.get_data_id(write_data_name, self._mesh_id)
            substep_time = write_waveform._temporal_grid[substep]
            write_data = write_waveform._samples_in_time[:,substep]
            super().write_block_scalar_data(write_data_id, self._n_vertices, self._vertex_ids, write_data)
            print("writing at time {time}".format(time=substep_time))
            print("write data called {name}:{write_data}".format(name=write_data_name, write_data=write_data))

    def _read_all_window_data_from_precice(self):
        read_data_name_prefix = self._read_data_name
        read_waveform = self._read_data_buffer
        init_data, init_time = read_waveform.get_init()
        read_waveform.empty_data()
        read_waveform.append(init_data, init_time)
        read_times = np.linspace(self._current_window_start, self._current_window_end(), self._n_other + 1)  # todo THIS IS HARDCODED! FOR ADAPTIVE GRIDS THIS IS NOT FITTING.
        for substep in range(self._n_other):
            read_data_name = read_data_name_prefix + str(substep)
            read_data_id = self.get_data_id(read_data_name, self._mesh_id)
            read_data = init_data.copy()
            substep_time = read_times[substep + 1]
            super().read_block_scalar_data(read_data_id, self._n_vertices, self._vertex_ids, read_data)
            print("reading at time {time}".format(time=substep_time))
            print("read_data called {name}:{read_data}".format(name=read_data_name, read_data=read_data))
            read_waveform.append(read_data, substep_time)

    def advance(self, dt):
        self._window_time += dt

        if self._window_is_completed():
            print("WINDOW COMPLETE!")
            self._write_all_window_data_to_precice()
            max_dt = super().advance(self._window_time)  # = time given by preCICE
            self._read_all_window_data_from_precice()

            # checkpointing
            if self.is_action_required(action_read_iteration_checkpoint()):
                # repeat window
                print("REPEAT")
                self._window_time = 0
                pass
            else:
                print("NEXT")
                # go to next window
                write_data_init = self._write_data_buffer.sample(self._current_window_end()).copy()
                read_data_init = self._read_data_buffer.sample(self._current_window_end()).copy()
                print("write_data_init with {write_data} from t = {time}".format(write_data=write_data_init, time=self._current_window_end()))
                self._current_window_start += self._window_size()
                self._window_time = 0
                self._write_data_buffer = self._get_empty_write_buffer(write_data_init)
                self._read_data_buffer = self._get_empty_read_buffer(read_data_init)
                self._print_window_status()

        else:
            print("remaining time: {remain}".format(remain=self._remaining_window_time()))
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
        if np.isclose(self._window_size(), self._window_time):
            print("COMPLETE!")
            return True
        else:
            return False

    def _remaining_window_time(self):
        return self._window_size() - self._window_time

    def _current_window_end(self):
        return self._current_window_start + self._window_size()

    def _is_inside_current_window(self, time):
        tol = self._window_size() * 10**-5
        return 0-tol <= time - self._current_window_start <= self._window_size() + tol

    def _window_size(self):
        return self._precice_tau

    def initialize(self):
        self._precice_tau = super().initialize_constant()
        return np.max([self._precice_tau, self._remaining_window_time()])

    def initialize_data(self):
        return super().initialize_data()

    def _do_interpolation(self, data, window_time):
        # this is currently a very limited dummy implementation

        # todo support "real" multirate, then remove following assertion
        assert(self._n_this == self._n_other)  # if self._N_this == self._N_other, we can assume that self._write_data = self._read_data and do not have to interpolate

        # todo support sampling data at arbitrary times
        assert(window_time * self._n_this % self._window_size() == 0)  # sampling time is exactly aligned with substep

        id_sample_at = round(window_time / self._window_size() * self._n_this)

        return data[id_sample_at]


class OutOfLocalWindowError(Exception):
    """Raised when the time is not inside the window; i.e. t not inside [t_start, t_end]"""
    pass


class NotOnTemporalGridError(Exception):
    """Raised when the point in time is not on the temporal grid. """
    pass


class NoDataError(Exception):
    """Raised if not data exists in waveform"""
    pass


class Waveform:
    def __init__(self, window_start, window_size, n_datapoints):
        """
        :param window_start: starting time of the window
        :param window_size: size of window
        :param n_samples: number of samples on window
        """
        assert (n_datapoints >= 1)
        assert (window_size > 0)

        self._window_size = window_size
        self._window_start = window_start
        self._n_datapoints = n_datapoints
        self._samples_in_time = None
        self._temporal_grid = None
        self.empty_data()

    def _window_end(self):
        return self._window_start + self._window_size

    def _append_sample(self, data, time):
        """
        appends a new piece of data for given time to the datastructures
        :param data: new dataset
        :param time: associated time
        :return:
        """
        data = data.reshape((data.size,1))
        print("###")
        print(self._samples_in_time.shape)
        print(data.shape)
        print("###")
        self._samples_in_time = np.append(self._samples_in_time, data, axis=1)
        self._temporal_grid.append(time)

    def initialize_constant(self, data):
        assert (not self._temporal_grid)  # list self._temporal_grid is empty
        assert (self._samples_in_time.size == 0)  # numpy.array self._samples_in_time is empty

        self._append_sample(data, self._window_start)
        self._append_sample(data, self._window_end())

    def sample(self, time):
        from scipy.interpolate import interp1d
        print("sample Waveform at %f" % time)

        if not self._temporal_grid:
            raise NoDataError

        atol = 1e-08  # todo: this is equal to atol used by default in np.isclose. Is there a nicer way to implement the check below?
        if not (np.min(self._temporal_grid) - atol <= time <= np.max(self._temporal_grid) + atol):
            msg = "\ntime: {time} on temporal grid {grid}\n".format(
                time=time,
                grid=self._temporal_grid)
            raise OutOfLocalWindowError(msg)

        return_value = np.zeros(self._n_datapoints)
        for i in range(self._n_datapoints):
            values_along_time = dict()
            for j in range(len(self._temporal_grid)):
                t = self._temporal_grid[j]
                print(self._samples_in_time)
                values_along_time[t] = self._samples_in_time[i, j]
            interpolant = interp1d(list(values_along_time.keys()), list(values_along_time.values()))
            try:
                return_value[i] = interpolant(time)
            except ValueError:
                time_min = np.min(self._temporal_grid)
                time_max = np.max(self._temporal_grid)

                if not time_min <= time <= time_max:  # time is not in valid range [time_min,time_max]
                    atol = 10**-8
                    if time_min-atol <= time <= time_min:  # time < time_min within within tolerance atol -> truncuate
                        time = time_min
                    elif time_max <= time <= time_max+atol:  # time > time_max within within tolerance atol -> truncuate
                        time = time_max
                    else:
                        raise Exception("Invalid time {time} computed!".format(time=time))
                return_value[i] = interpolant(time)

        return return_value

    def append(self, data, time):
        assert (data.shape[0] == self._n_datapoints)
        if time in self._temporal_grid or (self._temporal_grid and time <= self._temporal_grid[-1]):
            raise Exception("It is only allowed to append data associated with time that is larger than the already existing time. Trying to append invalid time = {time} to temporal grid = {temporal_grid}".format(time=time, temporal_grid=self._temporal_grid))
        self._append_sample(data, time)

    def empty_data(self):
        self._samples_in_time = np.empty(shape=(self._n_datapoints, 0))  # store samples in time in this data structure. Number of rows = number of gridpoints per sample; number of columns = number of sampls in time
        self._temporal_grid = list()  # store time associated to samples in this datastructure

    def get_init(self):
        return self._samples_in_time[:,0], self._temporal_grid[0]

