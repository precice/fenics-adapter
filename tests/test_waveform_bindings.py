# comments on test layout: https://docs.pytest.org/en/latest/goodpractices.html
# first install package python setup.py install
# then run tests with python setup.py test -s tests.test_waveform_bindings

from unittest.mock import MagicMock, patch
from unittest import TestCase
import warnings
import numpy as np
import numpy.testing as npt
import tests.MockedPySolverInterface

fake_dolfin = MagicMock()

@patch.dict('sys.modules', **{'dolfin': fake_dolfin, 'PySolverInterface': tests.MockedPySolverInterface})
class TestWaveformBindings(TestCase):

    dt = 1
    t = 0
    n = 0
    dummy_config_WR = "tests/precice-adapter-config-WR.json"
    n_data = 1
    n_vertices = 5

    def setUp(self):
        warnings.simplefilter('ignore', category=ImportWarning)

    def test_import(self):
        from fenicsadapter.waveform_bindings import WaveformBindings

    def test_init_fail(self):
        from fenicsadapter.waveform_bindings import WaveformBindings
        try:
            WaveformBindings()  # -> should throw a TypeError
            self.assertTrue(False)
        except Exception as e:
            self.assertEqual(type(e), TypeError)

    def test_init(self):
        from fenicsadapter.waveform_bindings import WaveformBindings
        WaveformBindings("Dummy", 0, 1)

    def test_read(self):
        from fenicsadapter.waveform_bindings import WaveformBindings
        from PySolverInterface import PySolverInterface

        def readBehavior(read_data_id, n_vertices, vertex_ids, read_data):
            assert (type(read_data) == np.ndarray)
            read_data += 1

        PySolverInterface.getDataID = MagicMock()
        PySolverInterface.readBlockScalarData = MagicMock(side_effect=readBehavior)
        bindings = WaveformBindings("Dummy", 0, 1)
        bindings.configure_waveform_relaxation(self.dummy_config_WR)
        bindings._precice_tau = self.dt
        read_data = np.zeros(self.n_vertices)
        dummy_mesh_id = MagicMock()
        dummy_vertex_ids = MagicMock()
        to_be_read = np.random.rand(self.n_vertices)
        bindings.initialize_waveforms(dummy_mesh_id, self.n_vertices, dummy_vertex_ids, "Dummy-Write", "Dummy-Read", self.n_data)
        bindings._read_data_buffer[0] = to_be_read
        bindings.readBlockScalarData("Dummy-Read", dummy_mesh_id, self.n_vertices, dummy_vertex_ids, read_data, 0)
        self.assertTrue(np.isclose(read_data, to_be_read).all())

    def test_write(self):
        from fenicsadapter.waveform_bindings import WaveformBindings
        from PySolverInterface import PySolverInterface

        PySolverInterface.getDataID = MagicMock()
        PySolverInterface.writeBlockScalarData = MagicMock()
        bindings = WaveformBindings("Dummy", 0, 1)
        bindings.configure_waveform_relaxation(self.dummy_config_WR)
        bindings._precice_tau = self.dt
        to_be_written = np.random.rand(self.n_vertices)
        write_data = to_be_written
        dummy_mesh_id = MagicMock()
        dummy_vertex_ids = MagicMock()
        bindings.initialize_waveforms(dummy_mesh_id, self.n_vertices, dummy_vertex_ids, "Dummy-Write", "Dummy-Read", self.n_data)
        bindings._write_data_buffer[0] = MagicMock()
        bindings.writeBlockScalarData("Dummy-Write", dummy_mesh_id, self.n_vertices, dummy_vertex_ids, write_data, 0)
        self.assertTrue(np.isclose(to_be_written, bindings._write_data_buffer).all())

    def test_do_some_steps(self):
        from fenicsadapter.waveform_bindings import WaveformBindings
        from PySolverInterface import PySolverInterface, PyActionReadIterationCheckpoint, \
            PyActionWriteIterationCheckpoint

        PySolverInterface.advance = MagicMock()
        PySolverInterface.getDataID = MagicMock()
        PySolverInterface.readBlockScalarData = MagicMock()
        PySolverInterface.writeBlockScalarData = MagicMock()
        bindings = WaveformBindings("Dummy", 0, 1)
        bindings.configure_waveform_relaxation(self.dummy_config_WR)
        dummy_mesh_id = MagicMock()
        dummy_vertex_ids = MagicMock()
        bindings.initialize_waveforms(dummy_mesh_id, self.n_vertices, dummy_vertex_ids, "Dummy-Write", "Dummy-Read", self.n_data)
        bindings._precice_tau = self.dt
        PySolverInterface.isActionRequired = MagicMock(return_value=False)
        self.assertEqual(bindings._current_window_start, 0.0)
        bindings.advance(.5)
        self.assertEqual(bindings._current_window_start, 0.0)
        bindings.advance(.5)
        self.assertEqual(bindings._current_window_start, 1.0)

        def isActionRequiredBehavior(py_action):
            if py_action == PyActionReadIterationCheckpoint():
                return True
            elif py_action == PyActionWriteIterationCheckpoint():
                return False
        PySolverInterface.isActionRequired = MagicMock(side_effect=isActionRequiredBehavior)
        bindings.advance(.5)
        self.assertEqual(bindings._current_window_start, 1.0)
        bindings.advance(.5)
        self.assertEqual(bindings._current_window_start, 1.0)

    def test_perform_substep(self):
        from fenicsadapter.waveform_bindings import WaveformBindings
        bindings = WaveformBindings("Dummy", 0, 1)
        bindings.configure_waveform_relaxation(self.dummy_config_WR)
        u0 = MagicMock(name="u0")
        u1 = MagicMock(name="u1")
        u1new = MagicMock(name="u1_new")
        v0 = MagicMock(name="v0")
        v1 = MagicMock(name="v1")

        bindings._write_data = [u0, u1]
        bindings._read_data = [v0, v1]

        bindings._perform_substep(u1new, self.t, self.dt, self.n)
        # TODO as soon as the functionality of the bindings gets more concrete, reactivate these assertions
        """
        self.assertEqual(bindings._write_data[0], u0)
        self.assertEqual(bindings._write_data[1], u1new)
        self.assertEqual(bindings._read_data[0], v0)
        self.assertEqual(bindings._read_data[1], v1)
        """

@patch.dict('sys.modules', **{'dolfin': fake_dolfin, 'PySolverInterface': tests.MockedPySolverInterface})
class TestWaveform(TestCase):
    def setUp(self):
        warnings.simplefilter('ignore', category=ImportWarning)

    def test_import(self):
        from fenicsadapter.waveform_bindings import Waveform

    def test_init(self):
        from fenicsadapter.waveform_bindings import Waveform
        wf = Waveform(np.linspace(0,1,10),2,3)

    def test_initialize_data(self):
        from fenicsadapter.waveform_bindings import Waveform, OutOfWindowError
        window_start = 2
        window_size = 3
        wf = Waveform(np.linspace(0,1,10),window_start,window_size)
        input_data = np.array([1,2,3])
        wf.initialize(input_data)
        for t in np.linspace(window_start, window_start + window_size):
            out = wf.sample(t)
            npt.assert_almost_equal(out, input_data)

        with self.assertRaises(OutOfWindowError):
            wf.sample(window_start + window_size + .1)
        with self.assertRaises(OutOfWindowError):
            wf.sample(window_start - .1)

    def test_update_data(self):
        from fenicsadapter.waveform_bindings import Waveform, OutOfWindowError, NotInTemporalGridError
        window_start = 2
        window_size = 3
        local_time_grid = np.linspace(0,1,10)
        global_time_grid = window_start + window_size * local_time_grid
        wf = Waveform(local_time_grid, window_start, window_size)
        input_data = np.array([1,2,3])
        wf.initialize(input_data)
        wf.update(input_data*2, global_time_grid[0])

        with self.assertRaises(NotInTemporalGridError):
            wf.update(input_data*2, local_time_grid[0])

        out = wf.sample(global_time_grid[0])
        npt.assert_almost_equal(out, input_data*2)

        for t in global_time_grid[1:]:
            out = wf.sample(t)
            npt.assert_almost_equal(out, input_data)

        out = wf.sample(.5 * (global_time_grid[0] + global_time_grid[1]))
        npt.assert_almost_equal(out, input_data*1.5)
