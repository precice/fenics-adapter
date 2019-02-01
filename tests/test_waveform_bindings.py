# comments on test layout: https://docs.pytest.org/en/latest/goodpractices.html
# first install package python setup.py install
# then run tests with python setup.py test -s tests.test_waveform_bindings

from unittest.mock import MagicMock, patch, Mock
from unittest import TestCase
import warnings
import numpy as np

fake_dolfin = MagicMock()
fake_PySolverInterface = MagicMock()


@patch.dict('sys.modules', **{'PySolverInterface': fake_PySolverInterface, 'dolfin': fake_dolfin})
class TestWaveformBindings(TestCase):

    dt = 1
    t = 0
    n = 0
    dummy_config_WR = "tests/precice-adapter-config-WR.json"

    def setUp(self):
        fake_PySolverInterface.PyActionReadIterationCheckpoint = MagicMock(return_value=1)
        fake_PySolverInterface.PyActionWriteIterationCheckpoint = MagicMock(return_value=2)
        warnings.simplefilter('ignore', category=ImportWarning)

    def mock_the_class(self):
        mocked_class = Mock

        def read_behavior(self, read_data_id, n_vertices, vertex_ids, read_data):
            assert(type(read_data) == np.ndarray)
            read_data += 1
            pass

        def write_behavior(self, write_data_id, n_vertices, vertex_ids, write_data):
            assert(type(write_data) == np.ndarray)
            write_data += 2
            pass

        mocked_class.writeBlockScalarData = MagicMock(side_effect=write_behavior)
        mocked_class.readBlockScalarData = MagicMock(side_effect=read_behavior)
        mocked_class.advance = MagicMock(return_value=self.dt)
        return mocked_class

    def test_import(self):
        from fenicsadapter.waveform_bindings import WaveformBindings

    def test_init(self):
        from fenicsadapter.waveform_bindings import WaveformBindings
        WaveformBindings()

    def test_read(self):
        from fenicsadapter.waveform_bindings import WaveformBindings
        fake_PySolverInterfaceClass = self.mock_the_class()
        import fenicsadapter
        with patch('PySolverInterface.PySolverInterface') as fake_PySolverInterfaceClass:
            bindings = WaveformBindings()
            n = 5
            read_data = np.zeros(n)
            bindings.readBlockScalarData("dummy", None, None, None, read_data)
            self.assertTrue(np.isclose(read_data, np.ones(n)).all())

    def test_write(self):
        from fenicsadapter.waveform_bindings import WaveformBindings
        fake_PySolverInterfaceClass = self.mock_the_class()
        import fenicsadapter
        with patch('PySolverInterface.PySolverInterface') as fake_PySolverInterfaceClass:
            bindings = WaveformBindings()
            n = 5
            write_data = np.zeros(n)
            bindings.writeBlockScalarData("dummy", None, None, None, write_data)
            self.assertTrue(np.isclose(write_data, 2*np.ones(n)).all())

    def test_perform_substep(self):
        fake_PySolverInterfaceClass = self.mock_the_class()
        import fenicsadapter
        with patch('PySolverInterface.PySolverInterface') as fake_PySolverInterfaceClass:
            from fenicsadapter.waveform_bindings import WaveformBindings
            bindings = WaveformBindings()

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
