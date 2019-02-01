# comments on test layout: https://docs.pytest.org/en/latest/goodpractices.html
# first install package python setup.py install
# then run tests with python setup.py test -s tests.test_waveform_bindings

from unittest.mock import MagicMock, patch
from unittest import TestCase
import warnings
import numpy as np

fake_dolfin = MagicMock()

@patch.dict('sys.modules', **{'dolfin': fake_dolfin})
class TestWaveformBindings(TestCase):

    dt = 1
    t = 0
    n = 0
    dummy_config_WR = "tests/precice-adapter-config-WR.json"

    def setUp(self):
        warnings.simplefilter('ignore', category=ImportWarning)

    def mock_the_class(self):
        mocked_class = MagicMock

        def read_behavior(self, read_data_id, n_vertices, vertex_ids, read_data, t):
            assert(type(read_data) == np.ndarray)
            read_data += 1
            pass

        def write_behavior(self, write_data_id, n_vertices, vertex_ids, write_data, t):
            assert(type(write_data) == np.ndarray)
            write_data += 2
            pass

        mocked_class.writeBlockScalarData = MagicMock(side_effect=write_behavior)
        mocked_class.readBlockScalarData = MagicMock(side_effect=read_behavior)
        mocked_class.advance = MagicMock(return_value=self.dt)
        mocked_class.initialize = MagicMock(return_value=self.dt)
        mocked_class.getDataID = MagicMock(return_value=0)
        return mocked_class

    def test_import(self):
        from fenicsadapter.waveform_bindings import WaveformBindings

    def test_init(self):
        from fenicsadapter.waveform_bindings import WaveformBindings
        with patch("PySolverInterface.PySolverInterface") as MagicMock:
            WaveformBindings("Dummy", 0, 1, self.dummy_config_WR)

    def test_read(self):
        from fenicsadapter.waveform_bindings import WaveformBindings
        fake_PySolverInterfaceClass = self.mock_the_class()
        with patch('PySolverInterface.PySolverInterface') as fake_PySolverInterfaceClass:
            bindings = WaveformBindings("Dummy", 0, 1, self.dummy_config_WR)
            bindings._precice_tau = self.dt
            n = 5
            read_data = np.zeros(n)
            bindings.readBlockScalarData("Dummy-Read", 0, None, None, read_data, 0)
            self.assertTrue(np.isclose(read_data, np.ones(n)).all())

    def test_write(self):
        from fenicsadapter.waveform_bindings import WaveformBindings
        fake_PySolverInterfaceClass = self.mock_the_class()
        with patch('PySolverInterface.PySolverInterface') as fake_PySolverInterfaceClass:
            bindings = WaveformBindings("Dummy", 0, 1, self.dummy_config_WR)
            bindings._precice_tau = self.dt
            print(bindings._config.get_write_data_name())
            n = 5
            write_data = np.zeros(n)
            bindings.writeBlockScalarData("Dummy-Write", 0, None, None, write_data, 0)
            self.assertTrue(np.isclose(write_data, 2*np.ones(n)).all())

    def test_perform_substep(self):
        from fenicsadapter.waveform_bindings import WaveformBindings
        fake_PySolverInterfaceClass = self.mock_the_class()
        with patch('PySolverInterface.PySolverInterface') as fake_PySolverInterfaceClass:
            bindings = WaveformBindings("Dummy", 0, 1, self.dummy_config_WR)

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