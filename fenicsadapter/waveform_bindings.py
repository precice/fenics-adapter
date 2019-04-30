import numpy as np
import logging

logging.basicConfig(level=logging.INFO)


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
        self._n_this = self._config.get_n_substeps()  # number of timesteps in this window, by default: no WR
        self._n_other = self._other_config.get_n_substeps()  # number of timesteps in other window, todo: in the end we don't want to worry about the other solver's resolution!
        self._current_window_start = 0  # defines start of window
        self._window_time = self._current_window_start  # keeps track of window time

    def initialize_waveforms(self, mesh_id, n_vertices, vertex_ids, write_data_name, read_data_name):
        logging.debug("Calling initialize_waveforms")
        logging.debug("Initializing waveforms.")
        # constant information of mesh
        self._mesh_id = mesh_id
        self._n_vertices = n_vertices
        self._vertex_ids = vertex_ids

        logging.debug("Creating write_data_buffer")
        self._write_data_name = write_data_name
        self._write_data_buffer = Waveform(self._current_window_start, self._precice_tau, self._n_vertices)

        logging.debug("Creating read_data_buffer")
        self._read_data_name = read_data_name
        self._read_data_buffer = Waveform(self._current_window_start, self._precice_tau, self._n_vertices)

    def write_block_scalar_data(self, write_data_name, mesh_id, n_vertices, vertex_ids, write_data, time):
        logging.debug("calling write_block_scalar_data for time {time}".format(time=time))
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
        logging.debug("calling read_block_scalar_data for time {time}".format(time=time))
        assert(self._config.get_read_data_name() == read_data_name)
        assert(self._is_inside_current_window(time))
        # we get the data from the interpolant. New data will be obtained from the other participant via preCICE in advance
        logging.debug("read at time {time}".format(time=time))
        read_data[:] = self._read_data_buffer.sample(time)[:].copy()
        logging.debug("read_data is {read_data}".format(read_data=read_data))
        # we assert that the preCICE specific write parameters did not change since configure_waveform_relaxation
        assert (self._mesh_id == mesh_id)
        assert (self._n_vertices == n_vertices)
        assert ((self._vertex_ids == vertex_ids).all())
        assert (self._read_data_name == read_data_name)

    def _write_all_window_data_to_precice(self):
        logging.debug("Calling _write_all_window_data_to_precice")
        write_data_name_prefix = self._write_data_name
        write_waveform = self._write_data_buffer
        logging.debug("write_waveform._temporal_grid = {grid}".format(grid=write_waveform._temporal_grid))
        for substep in range(1, self._n_this + 1):
            logging.debug("writing substep {substep} of {n_this}".format(substep=substep, n_this=self._n_this))
            write_data_name = write_data_name_prefix + str(substep)
            write_data_id = self.get_data_id(write_data_name, self._mesh_id)
            substep_time = write_waveform._temporal_grid[substep]
            write_data = write_waveform._samples_in_time[:, substep]
            super().write_block_scalar_data(write_data_id, self._n_vertices, self._vertex_ids, write_data)
            logging.debug("write data called {name}:{write_data} @ time = {time}".format(name=write_data_name,
                                                                                         write_data=write_data,
                                                                                         time=substep_time))

    def _rollback_write_data_buffer(self):
        self._write_data_buffer.empty_data(keep_first_sample=True)

    def _read_all_window_data_from_precice(self):
        logging.debug("Calling _read_all_window_data_from_precice")
        read_data_name_prefix = self._read_data_name
        read_waveform = self._read_data_buffer
        read_ndarray = read_waveform.get_empty_ndarray()
        read_waveform.empty_data(keep_first_sample=True)
        read_times = np.linspace(self._current_window_start, self._current_window_end(), self._n_other + 1)  # todo THIS IS HARDCODED! FOR ADAPTIVE GRIDS THIS IS NOT FITTING.

        for substep in range(1, self._n_other + 1):
            read_data_name = read_data_name_prefix + str(substep)
            read_data_id = self.get_data_id(read_data_name, self._mesh_id)
            read_data = np.copy(read_ndarray)
            substep_time = read_times[substep]
            super().read_block_scalar_data(read_data_id, self._n_vertices, self._vertex_ids, read_data)
            logging.debug("reading at time {time}".format(time=substep_time))
            logging.debug("read_data called {name}:{read_data} @ time = {time}".format(name=read_data_name,
                                                                                       read_data=read_data,
                                                                                       time=substep_time))
            read_waveform.append(read_data, substep_time)

    def advance(self, dt):
        self._window_time += dt

        if self._window_is_completed():
            logging.debug("Window is complete.")
            logging.debug("print read waveform")
            logging.debug(self._read_data_name)
            self._read_data_buffer.print_waveform()
            logging.debug("print write waveform")
            logging.debug(self._write_data_name)
            self._write_data_buffer.print_waveform()

            self._write_all_window_data_to_precice()
            logging.debug("calling precice.advance")
            read_data_last = self._read_data_buffer.sample(self._current_window_end()).copy()  # store last read data before advance, otherwise it might be lost if window is finished
            max_dt = super().advance(self._window_time)  # = time given by preCICE
            self._read_all_window_data_from_precice()

            logging.debug("print read waveform")
            logging.debug(self._read_data_name)
            self._read_data_buffer.print_waveform()
            logging.debug("print write waveform")
            logging.debug(self._write_data_name)
            self._write_data_buffer.print_waveform()

            if self.is_action_required(action_read_iteration_checkpoint()):  # repeat window
                # repeat window
                logging.debug("Repeat window.")
                self._rollback_write_data_buffer()
                self._window_time = 0
                pass
            else:  # window is finished
                logging.debug("Next window.")
                # go to next window
                read_data_init = read_data_last
                write_data_init = self._write_data_buffer.sample(self._current_window_end()).copy()
                logging.debug("write_data_init with {write_data} from t = {time}".format(write_data=write_data_init,
                                                                                         time=self._current_window_end()))
                logging.debug("read_data_init with {read_data} from t = {time}".format(read_data=read_data_init,
                                                                                       time=self._current_window_end()))
                self._current_window_start += self._window_size()
                self._window_time = 0
                # initialize window start of new window with data from window end of old window
                self._write_data_buffer = Waveform(self._current_window_start, self._precice_tau, self._n_vertices)
                self._write_data_buffer.append(write_data_init, self._current_window_start)
                # use constant extrapolation as initial guess for read data
                self._read_data_buffer = Waveform(self._current_window_start, self._precice_tau, self._n_vertices)
                self._read_data_buffer.append(read_data_init, self._current_window_start)
                self._read_data_buffer.append(read_data_init, self._current_window_end())
                self._print_window_status()

        else:
            logging.debug("remaining time: {remain}".format(remain=self._remaining_window_time()))
            max_dt = self._remaining_window_time()  # = window time remaining
            assert(max_dt > 0)

        return max_dt

    def _print_window_status(self):
        logging.debug("## window status:")
        logging.debug(self._current_window_start)
        logging.debug(self._window_size())
        logging.debug(self._window_time)
        logging.debug("##")

    def _window_is_completed(self):
        if np.isclose(self._window_size(), self._window_time):
            logging.debug("COMPLETE!")
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
        self._precice_tau = super().initialize()
        return np.max([self._precice_tau, self._remaining_window_time()])

    def is_action_required(self, action):
        if action == precice.action_write_initial_data():
            return True  # if we use waveform relaxation, we require initial data for both participants to be able to fill the write buffers correctly
        elif action == precice.action_write_iteration_checkpoint() or action == precice.action_read_iteration_checkpoint():
            return super().is_action_required(action)
        else:
            raise Exception("unexpected action. %s", str(action))

    def fulfilled_action(self, action):
        if action == precice.action_write_initial_data():
            return None  # do not forward to precice. We have to check for this condition again in initialize_data
        elif action == precice.action_write_iteration_checkpoint() or action == precice.action_read_iteration_checkpoint():
            return super().fulfilled_action(action)  # forward to precice
        else:
            raise Exception("unexpected action. %s", str(action))

    def initialize_data(self, time=0, read_zero=None, write_zero=None):
        """

        :param time:
        :param read_zero: read data that should be used at the very beginning
        :param write_zero: write data that should be used at the very beginning
        :return:
        """
        logging.debug("Calling initialize_data")
        if super().is_action_required(precice.action_write_initial_data()):
            logging.info("writing in initialize_data()")
            for substep in range(1, self._n_this + 1):
                time = substep * self._precice_tau / self._n_this
                self._write_data_buffer.append(write_zero, time)
            self._write_all_window_data_to_precice()
            self._rollback_write_data_buffer()
            super().fulfilled_action(precice.action_write_initial_data())

        return_value = super().initialize_data()

        if self.is_read_data_available():
            logging.info("reading in initialize_data()")
            self._read_data_buffer.empty_data(keep_first_sample=False)
            if isinstance(read_zero, np.ndarray):
                self._read_data_buffer.append(read_zero, self._current_window_start)
            else:
                self._read_data_buffer.append(self._read_data_buffer.get_empty_ndarray(), self._current_window_start)
            self._read_all_window_data_from_precice()
            if not isinstance(read_zero, np.ndarray):
                self._read_data_buffer.copy_second_to_first()

        return return_value


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
        logging.debug("###")
        logging.debug(self._samples_in_time.shape)
        logging.debug(data.shape)
        logging.debug("###")
        self._samples_in_time = np.append(self._samples_in_time, data, axis=1)
        self._temporal_grid.append(time)

    def initialize_constant(self, data):
        assert (not self._temporal_grid)  # list self._temporal_grid is empty
        assert (self._samples_in_time.size == 0)  # numpy.array self._samples_in_time is empty

        self._append_sample(data, self._window_start)
        self._append_sample(data, self._window_end())

    def sample(self, time):
        from scipy.interpolate import interp1d
        logging.debug("sample Waveform at %f" % time)

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

        logging.debug("result is {result}.".format(result=return_value))
        return return_value

    def append(self, data, time):
        assert (data.shape[0] == self._n_datapoints)
        if time in self._temporal_grid or (self._temporal_grid and time <= self._temporal_grid[-1]):
            raise Exception("It is only allowed to append data associated with time that is larger than the already existing time. Trying to append invalid time = {time} to temporal grid = {temporal_grid}".format(time=time, temporal_grid=self._temporal_grid))
        self._append_sample(data, time)

    def empty_data(self, keep_first_sample=False):
        if keep_first_sample:
            first_sample, first_time = self.get_init()
            assert(first_time == self._window_start)
        self._samples_in_time = np.empty(shape=(self._n_datapoints, 0))  # store samples in time in this data structure. Number of rows = number of gridpoints per sample; number of columns = number of sampls in time
        self._temporal_grid = list()  # store time associated to samples in this datastructure
        if keep_first_sample:
            self._append_sample(first_sample, first_time)

    def get_init(self):
        return self._samples_in_time[:, 0], self._temporal_grid[0]

    def get_empty_ndarray(self):
        return np.empty(self._n_datapoints)

    def copy_second_to_first(self):
        self._samples_in_time[:, 0] = self._samples_in_time[:, 1]

    def print_waveform(self):
        logging.debug("time: {time}".format(time=self._temporal_grid))
        logging.debug("data: {data}".format(data=self._samples_in_time))

