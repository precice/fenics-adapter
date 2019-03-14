# comments on test layout: https://docs.pytest.org/en/latest/goodpractices.html
# first install package python setup.py install
# then run tests with python setup.py test -s tests.test_waveform_bindings

from unittest.mock import MagicMock, patch
from unittest import TestCase
import warnings
import numpy as np
import tests.MockedPrecice

fake_dolfin = MagicMock()


@patch.dict('sys.modules', **{'dolfin': fake_dolfin, 'precice': tests.MockedPrecice})
class TestWaveformBindings(TestCase):

    dt = 1
    t = 0
    n = 0
    dummy_config = "tests/precice-adapter-config-WR10.json"
    n_vertices = 5

    def setUp(self):
        warnings.simplefilter('ignore', category=ImportWarning)

    def test_import(self):
        pass

    def test_init_fail(self):
        from fenicsadapter.waveform_bindings import WaveformBindings
        try:
            WaveformBindings()  # -> should throw a TypeError
            self.assertTrue(False)
        except Exception as e:
            self.assertEqual(type(e), TypeError)

    def test_init(self):
        with patch("precice.Interface") as tests.MockedPrecice.Interface:
            from fenicsadapter.waveform_bindings import WaveformBindings
            WaveformBindings("Dummy", 0, 1)

    def test_read(self):
        from fenicsadapter.waveform_bindings import WaveformBindings
        from precice import Interface

        def read_behavior(read_data_id, n_vertices, vertex_ids, read_data):
            assert (type(read_data) == np.ndarray)
            read_data += 1

        Interface.get_data_id = MagicMock()
        dummy_mesh_id = MagicMock()
        dummy_vertex_ids = np.random.rand(10)
        bindings = WaveformBindings("Dummy", 0, 1)
        bindings.configure_waveform_relaxation(self.dummy_config, self.dummy_config)
        bindings._precice_tau = self.dt
        old_data = np.random.rand(self.n_vertices)
        read_data = old_data
        to_be_read = old_data + 1
        bindings.initialize_waveforms(dummy_mesh_id, self.n_vertices, dummy_vertex_ids, "Dummy-Write", "Dummy-Read")
        bindings._read_data_buffer.update(to_be_read, 0)
        bindings.read_block_scalar_data("Dummy-Read", dummy_mesh_id, self.n_vertices, dummy_vertex_ids, read_data, 0)
        self.assertTrue(np.isclose(read_data, to_be_read).all())

    def test_write(self):
        from fenicsadapter.waveform_bindings import WaveformBindings
        from precice import Interface

        Interface.get_data_id = MagicMock()
        Interface.write_block_scalar_data = MagicMock()
        bindings = WaveformBindings("Dummy", 0, 1)
        bindings.configure_waveform_relaxation(self.dummy_config, self.dummy_config)
        bindings._precice_tau = self.dt
        dummy_mesh_id = MagicMock()
        dummy_vertex_ids = np.random.rand(10)
        old_data = np.random.rand(self.n_vertices)
        to_be_written = old_data + np.random.rand(self.n_vertices)
        write_data = to_be_written
        bindings.initialize_waveforms(dummy_mesh_id, self.n_vertices, dummy_vertex_ids, "Dummy-Write", "Dummy-Read")
        bindings._write_data_buffer.update(old_data, 0)
        bindings.write_block_scalar_data("Dummy-Write", dummy_mesh_id, self.n_vertices, dummy_vertex_ids, write_data, 0)
        self.assertTrue(np.isclose(to_be_written, bindings._write_data_buffer.sample(0)).all())

    def test_do_some_steps(self):
        from fenicsadapter.waveform_bindings import WaveformBindings
        from precice import Interface, action_read_iteration_checkpoint, action_write_iteration_checkpoint

        Interface.advance = MagicMock()
        Interface.get_data_id = MagicMock()
        Interface.read_block_scalar_data = MagicMock()
        Interface.write_block_scalar_data = MagicMock()
        bindings = WaveformBindings("Dummy", 0, 1)
        bindings.configure_waveform_relaxation(self.dummy_config, self.dummy_config)
        bindings._precice_tau = self.dt
        dummy_mesh_id = MagicMock()
        dummy_vertex_ids = np.random.rand(10)
        bindings.initialize_waveforms(dummy_mesh_id, self.n_vertices, dummy_vertex_ids, "Dummy-Write", "Dummy-Read")
        bindings._precice_tau = self.dt
        Interface.is_action_required= MagicMock(return_value=False)
        self.assertEqual(bindings._current_window_start, 0.0)
        for i in range(9):
            self.assertTrue(np.isclose(bindings._window_time, i * .1))
            bindings.advance(.1)
            self.assertTrue(np.isclose(bindings._window_time, (i+1) * .1))
            self.assertTrue(np.isclose(bindings._current_window_start, 0.0))
        bindings.advance(.1)
        self.assertTrue(np.isclose(bindings._current_window_start, 1.0))

        def is_action_required_behavior(py_action):
            if py_action == action_read_iteration_checkpoint():
                return True
            elif py_action == action_write_iteration_checkpoint():
                return False
        Interface.is_action_required = MagicMock(side_effect=is_action_required_behavior)
        for i in range(9):
            self.assertTrue(np.isclose(bindings._window_time, i * .1))
            bindings.advance(.1)
            self.assertTrue(np.isclose(bindings._window_time, (i+1) * .1))
            self.assertTrue(np.isclose(bindings._current_window_start, 1.0))
        bindings.advance(.1)
        self.assertTrue(np.isclose(bindings._current_window_start, 1.0))
