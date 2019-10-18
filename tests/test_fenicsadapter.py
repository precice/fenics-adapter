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

    def copy(self):
        returned_array = MockedArray()
        returned_array.value = self.value
        return returned_array

    def value_rank(self):
        return 0


@patch.dict('sys.modules', **{'dolfin': fake_dolfin, 'precice_future': tests.MockedPrecice})
class TestCheckpointing(TestCase):
    """
    Test suite to check whether checkpointing is done correctly, if the advance function is called. We use the mock
    pattern, for mocking the desired state of precice.
    """

    dt = 1  # timestep size
    n = 0  # current iteration count
    t = 0  # current time
    n_vertices = 10
    u_n_mocked = MockedArray()  # result at the beginning of the timestep
    u_np1_mocked = MockedArray()  # newly computed result
    write_function_mocked = MockedArray()
    u_cp_mocked = MockedArray()  # value of the checkpoint
    t_cp_mocked = t  # time for the checkpoint
    n_cp_mocked = n  # iteration count for the checkpoint
    dummy_config = "tests/precice-adapter-config-WR1.json"
    # todo if we support multirate, we should use the lines below for checkpointing
    # for the general case the checkpoint u_cp (and t_cp and n_cp) can differ from u_n and u_np1
    # t_cp_mocked = MagicMock()  # time for the checkpoint
    # n_cp_mocked = nMagicMock()  # iteration count for the checkpoint
    data_id = MagicMock()
    mesh_id = MagicMock()
    dummy_vertex_ids = np.arange(n_vertices)
    write_data_name = "Dummy-Write"
    read_data_name = "Dummy-Read"
    n_data = 10

    def setUp(self):
        warnings.simplefilter('ignore', category=ImportWarning)

    def mock_the_adapter(self, precice):
        from fenicsadapter.solverstate import SolverState
        from fenicsadapter import FunctionType
        """
        We partially mock the fenicsadapter, since proper configuration and initialization of the adapter is not
        necessary to test checkpointing.
        :param precice: the fenicsadapter
        """
        # define functions that are called by advance, but not necessary for the test
        precice._extract_coupling_boundary_coordinates = MagicMock(return_value=(None, None))
        precice._convert_fenics_to_precice = MagicMock(return_value=np.zeros(self.n_vertices))
        precice._interface._precice_tau = self.dt
        precice._coupling_bc_expression = MagicMock()
        precice._coupling_bc_expression.update_boundary_data = MagicMock()
        # initialize checkpointing manually
        mocked_state = SolverState(self.u_cp_mocked, self.t_cp_mocked, self.n_cp_mocked)
        precice._checkpoint.write(mocked_state)
        precice._precice_tau = 1
        precice._n_vertices = self.n_vertices
        precice._vertex_ids = self.dummy_vertex_ids
        precice._write_data_name = self.write_data_name
        precice._write_data = np.zeros(self.n_vertices)
        precice._read_data_name = self.read_data_name
        precice._read_data = np.zeros(self.n_vertices)
        precice._write_function_type = FunctionType.SCALAR
        precice._read_function_type = FunctionType.SCALAR

        from waveformbindings import WaveformBindings

        if type(precice._interface) is WaveformBindings:
            write_info = {"mesh_id": self.mesh_id, "n_vertices": self.n_vertices, "vertex_ids": self.dummy_vertex_ids,
                          "data_name": self.write_data_name, "data_dimension": 1}
            read_info = {"mesh_id": self.mesh_id, "n_vertices": self.n_vertices, "vertex_ids": self.dummy_vertex_ids,
                         "data_name": self.read_data_name, "data_dimension": 1}
            precice._interface.initialize_waveforms(write_info, read_info)

    def test_advance_success(self):
        """
        Test correct checkpointing, if advance succeeded
        """
        import fenicsadapter
        from precice_future import Interface

        Interface.reading_checkpoint_is_required = MagicMock(return_value=False)
        Interface.writing_checkpoint_is_required = MagicMock(return_value=True)
        Interface.configure = MagicMock()
        Interface.get_dimensions = MagicMock()
        Interface.get_mesh_id = MagicMock(return_value=self.mesh_id)
        Interface.get_data_id = MagicMock(return_value=self.data_id)
        Interface.write_block_scalar_data = MagicMock()
        Interface.read_block_scalar_data = MagicMock()
        Interface.is_read_data_available = MagicMock(return_value=True)
        Interface.is_write_data_required = MagicMock(return_value=True)
        Interface.advance = MagicMock(return_value=self.dt)
        Interface.fulfilled_action = MagicMock()

        precice = fenicsadapter.Adapter(self.dummy_config, self.dummy_config)  # todo: how can we avoid requiring both configs, if we do not use waveform relaxation?
        self.mock_the_adapter(precice)

        value_u_np1 = self.u_np1_mocked.value

        precice_step_complete = True
        # time and iteration count should be increased by a successful call of advance
        desired_output = (self.t + self.dt, self.n + 1, precice_step_complete, self.dt)
        self.assertEqual(precice.advance(None, self.u_np1_mocked, self.u_n_mocked, self.t, self.dt, self.n),
                         desired_output)

        # we expect that self.u_n_mocked.value has been updated to self.u_np1_mocked.value
        self.assertEqual(self.u_n_mocked.value, self.u_np1_mocked.value)

        # we expect that the value of the checkpoint has been updated to value_u_np1
        self.assertEqual(precice._checkpoint.get_state().u.value, value_u_np1)

    def test_advance_rollback(self):
        """
        Test correct checkpointing, if advance did not succeed and we have to rollback
        """
        import fenicsadapter
        from precice_future import Interface

        Interface.reading_checkpoint_is_required = MagicMock(return_value=True)
        Interface.writing_checkpoint_is_required = MagicMock(return_value=False)
        Interface.configure = MagicMock()
        Interface.get_dimensions = MagicMock()
        Interface.get_mesh_id = MagicMock(return_value=self.mesh_id)
        Interface.get_data_id = MagicMock(return_value=self.data_id)
        Interface.write_block_scalar_data = MagicMock()
        Interface.read_block_scalar_data = MagicMock()
        Interface.is_read_data_available = MagicMock(return_value=True)
        Interface.is_write_data_required = MagicMock(return_value=True)
        Interface.advance = MagicMock(return_value=self.dt)
        Interface.fulfilled_action = MagicMock()

        precice = fenicsadapter.Adapter(self.dummy_config, self.dummy_config)  # todo: how can we avoid requiring both configs, if we do not use waveform relaxation?
        self.mock_the_adapter(precice)

        precice_step_complete = False
        # time and iteration count should be rolled back by a not successful call of advance
        desired_output = (self.t_cp_mocked, self.n_cp_mocked, precice_step_complete, self.dt)
        self.assertEqual(precice.advance(None, self.u_np1_mocked, self.u_n_mocked, self.t, self.dt, self.n),
                         desired_output)

        # we expect that self.u_n_mocked.value has been rolled back to self.u_cp_mocked.value
        self.assertEqual(self.u_n_mocked.value, self.u_cp_mocked.value)

        # we expect that the value of the checkpoint has not been updated
        self.assertEqual(precice._checkpoint.get_state().u.value, self.u_cp_mocked.value)

    def test_advance_continue(self):
        """
        Test correct checkpointing, if advance did succeed, but we do not write a checkpoint (for example, if we do subcycling)
        :param fake_PySolverInterface_PySolverInterface: mock instance of PySolverInterface.PySolverInterface
        """
        import fenicsadapter
        from precice_future import Interface
        print("__INIT__ ADAPTER")
        Interface.reading_checkpoint_is_required = MagicMock(return_value=False)
        Interface.writing_checkpoint_is_required = MagicMock(return_value=False)
        Interface.configure = MagicMock()
        Interface.get_dimensions = MagicMock()
        Interface.get_mesh_id = MagicMock(return_value=self.mesh_id)
        Interface.get_data_id = MagicMock(return_value=self.data_id)
        Interface.is_read_data_available = MagicMock(return_value=False)  # inside subcycling we do not write or read data
        Interface.is_write_data_required = MagicMock(return_value=False)
        Interface.advance = MagicMock(return_value=self.dt)
        Interface.fulfilled_action = MagicMock()
        print("__INIT__ ADAPTER")
        precice = fenicsadapter.Adapter(self.dummy_config, self.dummy_config)  # todo: how can we avoid requiring both configs, if we do not use waveform relaxation?
        print("__INIT__ ADAPTER DONE")
        self.mock_the_adapter(precice)
        precice_step_complete = False
        # time and iteration count should be rolled back by a not successful call of advance
        desired_output = (self.t + self.dt, self.n + 1, precice_step_complete, self.dt)
        self.assertEqual(precice.advance(None, self.u_np1_mocked, self.u_n_mocked, self.t, self.dt, self.n),
                         desired_output)

        # we expect that self.u_n_mocked.value has been updated to self.u_np1_mocked.value
        self.assertEqual(self.u_n_mocked.value, self.u_np1_mocked.value)

        # we expect that the value of the checkpoint has not been updated
        self.assertEqual(precice._checkpoint.get_state().u.value, self.u_cp_mocked.value)


@patch.dict('sys.modules', **{'dolfin': fake_dolfin, 'precice': tests.MockedPrecice})
class TestIsCouplingOngoing(TestCase):
    dummy_config = "tests/precice-adapter-config-WR10.json"

    def setUp(self):
        warnings.simplefilter('ignore', category=ImportWarning)

    def test_isCouplingOngoing(self):
        import fenicsadapter
        from precice_future import Interface
        Interface.is_coupling_ongoing = MagicMock(return_value=True)
        Interface.configure = MagicMock()
        Interface.get_dimensions = MagicMock()
        Interface.get_mesh_id = MagicMock()
        Interface.get_data_id = MagicMock()
        precice = fenicsadapter.Adapter(self.dummy_config, self.dummy_config)  # todo: how can we avoid requiring both configs, if we do not use waveform relaxation?

        self.assertEqual(precice.is_coupling_ongoing(), True)
