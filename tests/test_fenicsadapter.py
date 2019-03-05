# comments on test layout: https://docs.pytest.org/en/latest/goodpractices.html
# first install package python setup.py install
# then run tests with python setup.py test -s tests.test_fenicsadapter

from unittest.mock import MagicMock, patch
from unittest import TestCase
import warnings
import tests.MockedPrecice
import numpy as np

fake_dolfin = MagicMock()

class MockedArray:
    """
    mock of dolfin.Function
    """
    def __init__(self):
        self.value = MagicMock()

    def assign(self, new_value):
        """
        mock of dolfin.Function.assign
        :param new_value:
        :return:
        """
        self.value = new_value.value


@patch.dict('sys.modules', **{'dolfin': fake_dolfin, 'precice': tests.MockedPrecice})
class TestCheckpointing(TestCase):
    """
    Test suite to check whether checkpointing is done correctly, if the advance function is called. We use the mock
    pattern, for mocking the desired state of precice.
    """

    dt = 1  # timestep size
    n = 0  # current iteration count
    t = 0  # current time
    u_n_mocked = MockedArray()  # result at the beginning of the timestep
    u_np1_mocked = MockedArray()  # newly computed result
    u_cp_mocked = MockedArray()  # value of the checkpoint
    t_cp_mocked = t  # time for the checkpoint
    n_cp_mocked = n  # iteration count for the checkpoint
    dummy_config = "tests/precice-adapter-config.json"
    dummy_config_WR = "tests/precice-adapter-config-WR.json"
    # todo if we support multirate, we should use the lines below for checkpointing
    # for the general case the checkpoint u_cp (and t_cp and n_cp) can differ from u_n and u_np1
    # t_cp_mocked = MagicMock()  # time for the checkpoint
    # n_cp_mocked = nMagicMock()  # iteration count for the checkpoint
    mesh_id = MagicMock()
    n_vertices = 10
    vertex_ids = MagicMock()
    write_data_name = "Dummy-Write"
    read_data_name = "Dummy-Read"
    n_data = 1
    data_id = MagicMock()

    def setUp(self):
        warnings.simplefilter('ignore', category=ImportWarning)

    def mock_the_adapter(self, precice):
        """
        We partially mock the fenicsadapter, since proper configuration and initialization of the adapter is not
        necessary to test checkpointing.
        :param precice: the fenicsadapter
        """
        # define functions that are called by advance, but not necessary for the test
        precice.extract_coupling_boundary_coordinates = MagicMock(return_value=(None, None))
        precice.convert_fenics_to_precice = MagicMock()
        precice._interface._precice_tau = self.dt
        precice._coupling_bc_expression = MagicMock()
        precice._coupling_bc_expression.update_boundary_data = MagicMock()
        # initialize checkpointing manually
        precice._t_cp = self.t_cp_mocked
        precice._u_cp = self.u_cp_mocked
        precice._n_cp = self.n_cp_mocked
        precice._precice_tau = 1
        precice._n_vertices = self.n_vertices
        precice._vertex_ids = self.vertex_ids
        precice._write_data_name = self.write_data_name
        precice._write_data = np.zeros(self.n_vertices)
        precice._read_data_name = self.read_data_name
        precice._read_data = np.zeros(self.n_vertices)

        from fenicsadapter.waveform_bindings import WaveformBindings

        if type(precice._interface) is WaveformBindings:
            precice._interface.initialize_waveforms(self.mesh_id , self.n_vertices, self.vertex_ids, self.write_data_name, self.read_data_name, self.n_data)

    def test_advance_success(self):
        """
        Test correct checkpointing, if advance succeeded
        """
        import fenicsadapter
        from precice import Interface, action_read_iteration_checkpoint, \
            action_write_iteration_checkpoint

        def is_action_required_behavior(py_action):
            if py_action == action_read_iteration_checkpoint():
                return False
            elif py_action == action_write_iteration_checkpoint():
                return True

        Interface.is_action_required = MagicMock(side_effect=is_action_required_behavior)
        Interface.configure = MagicMock()
        Interface.get_dimensions = MagicMock()
        Interface.get_mesh_id = MagicMock(return_value=self.mesh_id)
        Interface.get_data_id = MagicMock(return_value=self.data_id)
        Interface.write_block_scalar_data = MagicMock()
        Interface.read_block_scalar_data = MagicMock()
        Interface.advance = MagicMock(return_value=self.dt)
        Interface.fulfilled_action = MagicMock()

        precice = fenicsadapter.Adapter(self.dummy_config)
        self.mock_the_adapter(precice)

        value_u_np1 = self.u_np1_mocked.value

        precice_step_complete = True
        # time and iteration count should be increased by a successful call of advance
        desired_output = (self.t + self.dt, self.n + 1, precice_step_complete, self.dt)
        self.assertEqual(precice.advance(None, self.u_np1_mocked, self.u_n_mocked, self.t, self.dt, self.n),
                         desired_output)

        # we expect that self.u_n_mocked.value has been updated to self.u_np1_mocked.value
        self.assertEqual(self.u_n_mocked.value, self.u_np1_mocked.value)

        # we expect that precice._u_cp.value has been updated to value_u_np1
        self.assertEqual(precice._u_cp.value, value_u_np1)

    def test_advance_rollback(self):
        """
        Test correct checkpointing, if advance did not succeed and we have to rollback
        """
        import fenicsadapter
        from precice import Interface, action_write_iteration_checkpoint, \
            action_read_iteration_checkpoint

        def is_action_required_behavior(py_action):
            if py_action == action_read_iteration_checkpoint():
                return True
            elif py_action == action_write_iteration_checkpoint():
                return False

        Interface.is_action_required = MagicMock(side_effect=is_action_required_behavior)
        Interface.configure = MagicMock()
        Interface.get_dimensions = MagicMock()
        Interface.get_mesh_id = MagicMock(return_value=self.mesh_id)
        Interface.get_data_id = MagicMock(return_value=self.data_id)
        Interface.write_block_scalar_data = MagicMock()
        Interface.read_block_scalar_data = MagicMock()
        Interface.advance = MagicMock(return_value=self.dt)
        Interface.fulfilled_action = MagicMock()

        precice = fenicsadapter.Adapter(self.dummy_config)
        self.mock_the_adapter(precice)

        precice_step_complete = False
        # time and iteration count should be rolled back by a not successful call of advance
        desired_output = (self.t_cp_mocked, self.n_cp_mocked, precice_step_complete, self.dt)
        self.assertEqual(precice.advance(None, self.u_np1_mocked, self.u_n_mocked, self.t, self.dt, self.n),
                         desired_output)

        # we expect that self.u_n_mocked.value has been rolled back to self.u_cp_mocked.value
        self.assertEqual(self.u_n_mocked.value, self.u_cp_mocked.value)

        # we expect that precice._u_cp.value has not been updated
        self.assertEqual(precice._u_cp.value, self.u_cp_mocked.value)

    def test_advance_continue(self):
        """
        Test correct checkpointing, if advance did succeed, but we do not write a checkpoint (for example, if we do subcycling)
        """
        import fenicsadapter
        from precice import Interface, action_read_iteration_checkpoint, \
            action_write_iteration_checkpoint

        def is_action_required_behavior(py_action):
            if py_action == action_read_iteration_checkpoint():
                return False
            elif py_action == action_write_iteration_checkpoint():
                return False

        Interface.is_action_required = MagicMock(side_effect=is_action_required_behavior)
        Interface.configure = MagicMock()
        Interface.get_dimensions = MagicMock()
        Interface.get_mesh_id = MagicMock(return_value=self.mesh_id)
        Interface.get_data_id = MagicMock(return_value=self.data_id)
        Interface.write_block_scalar_data = MagicMock()
        Interface.read_block_scalar_data = MagicMock()
        Interface.advance = MagicMock(return_value=self.dt)
        Interface.fulfilled_action = MagicMock()

        precice = fenicsadapter.Adapter(self.dummy_config)
        self.mock_the_adapter(precice)

        precice_step_complete = False
        # time and iteration count should be rolled back by a not successful call of advance
        desired_output = (self.t + self.dt, self.n + 1, precice_step_complete, self.dt)
        self.assertEqual(precice.advance(None, self.u_np1_mocked, self.u_n_mocked, self.t, self.dt, self.n),
                         desired_output)

        # we expect that self.u_n_mocked.value has been updated to self.u_np1_mocked.value
        self.assertEqual(self.u_n_mocked.value, self.u_np1_mocked.value)

        # we expect that precice._u_cp.value has not been updated
        self.assertEqual(precice._u_cp.value, self.u_cp_mocked.value)


@patch.dict('sys.modules', **{'dolfin': fake_dolfin, 'precice': tests.MockedPrecice})
class TestIsCouplingOngoing(TestCase):
    dummy_config = "tests/precice-adapter-config.json"

    def setUp(self):
        warnings.simplefilter('ignore', category=ImportWarning)

    def test_isCouplingOngoing(self):
        import fenicsadapter
        from precice import Interface
        Interface.is_coupling_ongoing = MagicMock(return_value=True)
        Interface.configure = MagicMock()
        Interface.get_dimensions = MagicMock()
        Interface.get_mesh_id = MagicMock()
        Interface.get_data_id = MagicMock()
        precice = fenicsadapter.Adapter(self.dummy_config)

        self.assertEqual(precice.is_coupling_ongoing(), True)
