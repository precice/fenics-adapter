# comments on test layout: https://docs.pytest.org/en/latest/goodpractices.html
# first install package python setup.py install
# then run tests with python setup.py test -s tests.test_waveform_bindings

from unittest.mock import MagicMock, patch
from unittest import TestCase
import warnings
import numpy as np
import tests.MockedPySolverInterface

fake_dolfin = MagicMock()

@patch.dict('sys.modules', **{'dolfin': fake_dolfin, 'PySolverInterface': tests.MockedPySolverInterface})
class TestWaveformBindings(TestCase):

    dt = 1
    t = 0
    n = 0
    dummy_config_WR = "tests/precice-adapter-config-WR.json"

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
        n = 5
        read_data = np.zeros(n)
        bindings.readBlockScalarData("Dummy-Read", 0, None, None, read_data, 0)
        self.assertTrue(np.isclose(read_data, np.ones(n)).all())

    def test_write(self):
        from fenicsadapter.waveform_bindings import WaveformBindings
        from PySolverInterface import PySolverInterface

        def writeBehavior(read_data_id, n_vertices, vertex_ids, read_data):
            assert (type(read_data) == np.ndarray)
            read_data += 2

        PySolverInterface.getDataID = MagicMock()
        PySolverInterface.writeBlockScalarData = MagicMock(side_effect=writeBehavior)
        bindings = WaveformBindings("Dummy", 0, 1)
        bindings.configure_waveform_relaxation(self.dummy_config_WR)
        bindings._precice_tau = self.dt
        n = 5
        write_data = np.zeros(n)
        bindings.writeBlockScalarData("Dummy-Write", 0, None, None, write_data, 0)
        self.assertTrue(np.isclose(write_data, 2*np.ones(n)).all())

    def test_do_some_steps(self):
        from fenicsadapter.waveform_bindings import WaveformBindings
        from PySolverInterface import PySolverInterface, PyActionReadIterationCheckpoint, \
            PyActionWriteIterationCheckpoint

        PySolverInterface.advance = MagicMock()
        bindings = WaveformBindings("Dummy", 0, 1)
        bindings.configure_waveform_relaxation(self.dummy_config_WR)
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
