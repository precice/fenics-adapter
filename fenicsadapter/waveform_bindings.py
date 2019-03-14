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


class WrongTimestepSizeError(Exception):
    pass

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
                             write_data_init=None, read_data_init=None):
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

    def _get_empty_write_buffer(self, write_data_init=None):
        return self._get_empty_buffer(self._n_this, write_data_init)

    def _get_empty_read_buffer(self, read_data_init=None):
        return self._get_empty_buffer(self._n_other, read_data_init)

    def _get_empty_buffer(self, n_substeps, data_init=None):
        if data_init is None:
            data_init = np.zeros(self._n_vertices)
        #print("creating new waveform! start @ {start}, width {tau}".format(start=self._current_window_start, tau=self._precice_tau))
        buffer = Waveform(self._current_window_start, self._precice_tau, n_substeps)
        buffer.initialize(data_init)
        return buffer

    def write_block_scalar_data(self, write_data_name, mesh_id, n_vertices, vertex_ids, write_data, time):
        assert(self._config.get_write_data_name() == write_data_name)
        assert(self._is_inside_current_window(time))
        # we put the data into a buffer. Data will be send to other participant via preCICE in advance
        print("write at time {time}".format(time=time))
        self._write_data_buffer.update(write_data[:], time)
        print("write_data is {write_data}".format(write_data=write_data[:]))
        print("write_data is {write_data}".format(write_data=self._write_data_buffer.sample(time)))
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
            substep_time = write_waveform.get_global_time(substep)
            write_data = write_waveform.sample(substep_time)
            super().write_block_scalar_data(write_data_id, self._n_vertices, self._vertex_ids, write_data)
            print("writing at time {time}".format(time=substep_time))
            print("write data called {name}:{write_data}".format(name=write_data_name, write_data=write_data))

    def _read_all_window_data_from_precice(self):
        read_data_name_prefix = self._read_data_name
        read_waveform = self._read_data_buffer
        for substep in range(self._n_other):
            read_data_name = read_data_name_prefix + str(substep)
            read_data_id = self.get_data_id(read_data_name, self._mesh_id)
            substep_time = read_waveform.get_global_time(substep)
            read_data = read_waveform.sample(substep_time)
            super().read_block_scalar_data(read_data_id, self._n_vertices, self._vertex_ids, read_data)
            print("reading at time {time}".format(time=substep_time))
            print("read_data called {name}:{read_data}".format(name=read_data_name, read_data=read_data))
            read_waveform.update(read_data, substep_time)

    def advance(self, dt):
        self._window_time += dt
        if not np.isclose(dt, self._write_data_buffer.get_dt()):
            msg = "Expected timestep size dt={dt_expected}. Received dt={dt_received}.".format(dt_expected=self._write_data_buffer.get_dt(), dt_received=dt)
            raise WrongTimestepSizeError(msg)

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

    def _is_inside_current_window(self, global_time):
        local_time = global_time - self._current_window_start
        tol = self._window_size() * 10**-5
        return 0-tol <= local_time <= self._window_size()+tol

    def _window_size(self):
        return self._precice_tau

    def initialize(self):
        self._precice_tau = super().initialize()
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
    def __init__(self, window_start, window_size, n_samples):
        """
        :param window_start: starting time of the window
        :param window_size: size of window
        :param n_samples: number of samples on window
        """
        assert (n_samples >= 2)
        assert (window_size > 0)
        self._temporal_grid, self._dt = np.linspace(0, 1, n_samples, retstep=True)
        self._samples_in_time = dict()
        self._window_size = window_size
        self._window_start = window_start
        self._n_datapoints = None
        for t in self._temporal_grid:
            self._samples_in_time[t] = None

    def get_local_time(self, grid_id):
        return self._temporal_grid[grid_id]

    def get_global_time(self, grid_id):
        return self._local_to_global_time(self.get_local_time(grid_id))

    def get_dt(self):
        # todo: currently, we assume a constant dt for a waveform. Generally, this is not the case and we should allow adaptive strategies!
        return self._get_dt() * self._window_size

    def _get_dt(self):
        return self._dt

    def initialize(self, data):
        self._n_datapoints = data.shape[0]
        for t in self._temporal_grid:
            self._samples_in_time[t] = data.copy()

    def _sample(self, local_time):
        from scipy.interpolate import interp1d
        print("sample Waveform at %f" % local_time)

        if not self._n_datapoints:
            raise NoDataError

        atol = 1e-08  # todo: this is equal to atol used by default in np.isclose. Is there a nicer way to implement the check below?
        if not (0 - atol <= local_time <= 1 + atol):
            msg = "\nlocal time: {time} on local grid {grid}\n" \
                  "corresponds to\n" \
                  "global time: {global_time} on global grid {ggrid}".format(
                time=local_time,
                grid=self._temporal_grid,
                global_time=self._local_to_global_time(local_time),
                ggrid=self.global_temporal_grid())
            raise OutOfLocalWindowError(msg)

        return_value = np.zeros(self._n_datapoints)
        for i in range(self._n_datapoints):
            values_along_time = dict()
            for t in self._temporal_grid:
                values_along_time[t] = self._samples_in_time[t][i]
            interpolant = interp1d(list(values_along_time.keys()), list(values_along_time.values()))
            if not (0 <= local_time <= 1) and (0 - atol <= local_time <= 1 + atol):  # local time is at the boundary of the interval within a given tolerance
                if local_time < 0:
                    local_time = 0
                elif local_time > 1:
                    local_time = 1
                else:
                    raise Exception("unexpected behavior!")
            return_value[i] = interpolant(local_time)
        return return_value

    def sample(self, global_time):
        local_time = self.global_to_local_time(global_time)
        return self._sample(local_time)

    def global_to_local_time(self, global_time):
        return (global_time - self._window_start)/self._window_size

    def _local_to_global_time(self, local_time):
        return (local_time * self._window_size) + self._window_start

    def global_temporal_grid(self):
        return self._temporal_grid * self._window_size + self._window_start

    def _time_is_on_grid(self, time):
        #print("Grid: {grid}".format(grid=self.global_temporal_grid()))
        return np.isclose(self.global_temporal_grid(), time).any()

    def _update(self, data, local_time):
        print("_update at {local_time}".format(local_time=local_time))
        if local_time in self._samples_in_time.keys():
            self._samples_in_time[local_time] = data
        else:
            key = self._temporal_grid[np.isclose(self._temporal_grid, local_time)][0]
            self._samples_in_time[key] = data
            #raise Exception("local_time {local_time} not found! Better use {key}".format(local_time=local_time, key=key))

    def update(self, data, global_time):
        #print("Global time: {global_time}".format(global_time=global_time))
        if not self._time_is_on_grid(global_time):
            msg = "trying to update at {global_time} while temporal grid is {grid}".format(global_time=global_time, grid=self.global_temporal_grid())
            raise NotOnTemporalGridError(msg)
        assert (data.shape[0] == self._n_datapoints)
        self._update(data, self.global_to_local_time(global_time))
