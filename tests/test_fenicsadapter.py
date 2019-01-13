# comments on test layout: https://docs.pytest.org/en/latest/goodpractices.html
# run with python -m unittest tests.test_fenicsadapter

from unittest.mock import MagicMock, patch
from unittest import TestCase
import warnings

fake_dolfin = MagicMock()
fake_PySolverInterface = MagicMock()


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


@patch.dict('sys.modules', **{'PySolverInterface': fake_PySolverInterface, 'dolfin': fake_dolfin})
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
    # todo if we support multirate, we should use the lines below for checkpointing
    # for the general case the checkpoint u_cp (and t_cp and n_cp) can differ from u_n and u_np1
    # t_cp_mocked = MagicMock()  # time for the checkpoint
    # n_cp_mocked = nMagicMock()  # iteration count for the checkpoint

    def setUp(self):
        fake_PySolverInterface.PyActionReadIterationCheckpoint = MagicMock(return_value=1)
        fake_PySolverInterface.PyActionWriteIterationCheckpoint = MagicMock(return_value=2)
        warnings.simplefilter('ignore', category=ImportWarning)

    def mock_the_adapter(self, precice):
        """
        We partially mock the fenicsadapter, since proper configuration and initialization of the adapter is not
        necessary to test checkpointing.
        :param precice: the fenicsadapter
        """
        precice.configure(None, None, None, None, None)
        # define functions that are called by advance, but not necessary for the test
        precice.extract_coupling_boundary_coordinates = MagicMock(return_value=(None, None))
        precice.convert_fenics_to_precice = MagicMock()
        precice._coupling_bc_expression = MagicMock()
        precice._coupling_bc_expression.update_boundary_data = MagicMock()
        precice._do_interpolation = MagicMock()
        precice._write_data = [None, None]
        precice._write_data_id = [None, None]
        precice._read_data = [None, None]
        precice._read_data_id = [None, None]
        precice._window_size = self.dt
        precice._N_this = 1  # number of timesteps in this window, by default: no WR
        precice._N_other = 1  # number of timesteps in other window
        precice._substep_counter = 0
        precice._precice_tau = self.dt
        # initialize checkpointing manually
        precice._t_cp = self.t_cp_mocked
        precice._u_cp = self.u_cp_mocked
        precice._n_cp = self.n_cp_mocked

    def mock_the_interface(self, interface, success):
        """
        We mock the PySolverInterface such that we can easily get the desired behavior when called by advance. If
        success is True, our mocked PySolverInterface behaves as if the current iteration was successful. If success is
        False, our mocked PySolverInterface behaves as if the current iteration was not successful and a checkpoint has
        to be loaded.
        :param interface: mocked PySolverInterface.PySolverInterface object where we add functionality needed for testing
        :param success: specify whether the interface should
        """
        import PySolverInterface

        def mocked_behavior(py_action):
            """
            Depending on py_action, this mocked function returns a bool.
            :param py_action: PySolverInterface.PyActionReadIterationCheckpoint or PySolverInterface.PyActionWriteIterationCheckpoint
            :return:
            """
            if py_action == PySolverInterface.PyActionReadIterationCheckpoint():
                return not success
            elif py_action == PySolverInterface.PyActionWriteIterationCheckpoint():
                return success

        """
        return value of isActionRequired is crucial for behavior of advance w.r.t checkpointing. 
        We mock it to directly get the behavior of advance that we want to test
        """
        interface.return_value.isActionRequired = MagicMock(side_effect=mocked_behavior)
        # functions below are called by advance and have to be defined, we are not interested in the output.
        interface.return_value.writeBlockScalarData = MagicMock()
        interface.return_value.readBlockScalarData = MagicMock()
        interface.return_value.advance = MagicMock()

    @patch('PySolverInterface.PySolverInterface')
    def test_advance_success(self, fake_PySolverInterface_PySolverInterface):
        """
        Test correct checkpointing, if advance succeeded
        :param fake_PySolverInterface_PySolverInterface: mock instance of PySolverInterface.PySolverInterface
        """
        success = True

        self.mock_the_interface(fake_PySolverInterface_PySolverInterface, success)

        import fenicsadapter
        precice = fenicsadapter.Adapter()
        self.mock_the_adapter(precice)

        value_u_np1 = self.u_np1_mocked.value

        # time and iteration count should be increased by a successful call of advance
        desired_output = (self.t + self.dt, self.n + 1, success)
        self.assertEqual(precice.advance(None, self.u_np1_mocked, self.u_n_mocked, self.t, self.dt, self.n),
                         desired_output)

        # we expect that self.u_n_mocked.value has been updated to self.u_np1_mocked.value
        self.assertEqual(self.u_n_mocked.value, self.u_np1_mocked.value)

        # we expect that precice._u_cp.value has been updated to value_u_np1
        self.assertEqual(precice._u_cp.value, value_u_np1)

    @patch('PySolverInterface.PySolverInterface')
    def test_advance_rollback(self, fake_PySolverInterface_PySolverInterface):
        """
        Test correct checkpointing, if advance did not succeed and we have to rollback
        :param fake_PySolverInterface_PySolverInterface: mock instance of PySolverInterface.PySolverInterface
        """
        success = False

        self.mock_the_interface(fake_PySolverInterface_PySolverInterface, success)

        import fenicsadapter
        precice = fenicsadapter.Adapter()
        self.mock_the_adapter(precice)

        # time and iteration count should be rolled back by a not successful call of advance
        desired_output = (self.t_cp_mocked, self.n_cp_mocked, success)
        self.assertEqual(precice.advance(None, self.u_np1_mocked, self.u_n_mocked, self.t, self.dt, self.n),
                         desired_output)

        # we expect that self.u_n_mocked.value has been rolled back to self.u_cp_mocked.value
        self.assertEqual(self.u_n_mocked.value, self.u_cp_mocked.value)

        # we expect that precice._u_cp.value has not been updated
        self.assertEqual(precice._u_cp.value, self.u_cp_mocked.value)


@patch.dict('sys.modules', **{'PySolverInterface': fake_PySolverInterface, 'dolfin': fake_dolfin})
class TestIsCouplingOngoing(TestCase):
    def setUp(self):
        fake_PySolverInterface.PyActionReadIterationCheckpoint = MagicMock(return_value=1)
        fake_PySolverInterface.PyActionWriteIterationCheckpoint = MagicMock(return_value=2)
        warnings.simplefilter('ignore', category=ImportWarning)

    @patch('PySolverInterface.PySolverInterface')
    def test_isCouplingOngoing(self, fake_PySolverInterface_PySolverInterface):
        import fenicsadapter
        precice = fenicsadapter.Adapter()
        precice.configure(None, None, None, None, None)

        fake_PySolverInterface_PySolverInterface.return_value.isCouplingOngoing = MagicMock(return_value=True)
        self.assertEqual(precice.is_coupling_ongoing(), True)

        fake_PySolverInterface_PySolverInterface.return_value.isCouplingOngoing = MagicMock(return_value=False)
        self.assertEqual(precice.is_coupling_ongoing(), False)
