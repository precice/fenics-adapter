# comments on test layout: https://docs.pytest.org/en/latest/goodpractices.html
# first install package python setup.py install
# then run tests with python setup.py test -s tests.test_fenicsadapter

from unittest.mock import MagicMock, patch, Mock
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
    dummy_config = "tests/precice-adapter-config.json"
    dummy_config_WR = "tests/precice-adapter-config-WR.json"
    # todo if we support multirate, we should use the lines below for checkpointing
    # for the general case the checkpoint u_cp (and t_cp and n_cp) can differ from u_n and u_np1
    # t_cp_mocked = MagicMock()  # time for the checkpoint
    # n_cp_mocked = nMagicMock()  # iteration count for the checkpoint

    def setUp(self):
        fake_PySolverInterface.PyActionReadIterationCheckpoint = MagicMock(return_value=1)
        fake_PySolverInterface.PyActionWriteIterationCheckpoint = MagicMock(return_value=2)
        warnings.simplefilter('ignore', category=ImportWarning)

    def mock_the_adapter(self, precice, mocked_class):
        """
        We partially mock the fenicsadapter, since proper configuration and initialization of the adapter is not
        necessary to test checkpointing.
        :param precice: the fenicsadapter
        """
        # define functions that are called by advance, but not necessary for the test
        precice._interface = mocked_class()
        precice.extract_coupling_boundary_coordinates = MagicMock(return_value=(None, None))
        precice.convert_fenics_to_precice = MagicMock()
        precice._coupling_bc_expression = MagicMock()
        precice._coupling_bc_expression.update_boundary_data = MagicMock()
        # initialize checkpointing manually
        precice._t_cp = self.t_cp_mocked
        precice._u_cp = self.u_cp_mocked
        precice._n_cp = self.n_cp_mocked
        precice._precice_tau = 1

    def mock_the_class(self, read_iteration_checkpoint_return=False, write_iteration_checkpoint_return=False):
        """
        We mock the PySolverInterface such that we can easily get the desired behavior when called by advance. If
        success is True, our mocked PySolverInterface behaves as if the current iteration was successful. If success is
        False, our mocked PySolverInterface behaves as if the current iteration was not successful and a checkpoint has
        to be loaded.
        :param read_iteration_checkpoint_return: specify whether the interface should read the existing checkpoint
        :param write_iteration_checkpoint_return: specify whether the interface should write a new checkpoint
        """
        import PySolverInterface

        mocked_class = Mock

        def mocked_behavior(py_action):
            """
            Depending on py_action, this mocked function returns a bool.
            :param py_action: PySolverInterface.PyActionReadIterationCheckpoint or PySolverInterface.PyActionWriteIterationCheckpoint
            :return:
            """
            if py_action == PySolverInterface.PyActionReadIterationCheckpoint():
                return read_iteration_checkpoint_return
            elif py_action == PySolverInterface.PyActionWriteIterationCheckpoint():
                return write_iteration_checkpoint_return

        """
        return value of isActionRequired is crucial for behavior of advance w.r.t checkpointing. 
        We mock it to directly get the behavior of advance that we want to test
        """
        mocked_class.isActionRequired = MagicMock(side_effect=mocked_behavior)
        # functions below are called by advance and have to be defined, we are not interested in the output.
        mocked_class.writeBlockScalarData = MagicMock()
        mocked_class.readBlockScalarData = MagicMock()
        mocked_class.advance = MagicMock(return_value=self.dt)
        return mocked_class

    def test_advance_success(self):
        """
        Test correct checkpointing, if advance succeeded
        """
        fake_PySolverInterfaceClass = self.mock_the_class(write_iteration_checkpoint_return=True)

        import fenicsadapter
        with patch('PySolverInterface.PySolverInterface') as fake_PySolverInterfaceClass:
            precice = fenicsadapter.Adapter(self.dummy_config)
            self.mock_the_adapter(precice, fake_PySolverInterfaceClass)

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
        fake_PySolverInterfaceClass = self.mock_the_class(read_iteration_checkpoint_return=True)

        import fenicsadapter
        with patch('PySolverInterface.PySolverInterface') as fake_PySolverInterfaceClass:
            precice = fenicsadapter.Adapter(self.dummy_config)
            self.mock_the_adapter(precice, fake_PySolverInterfaceClass)

            precice_step_complete = False
            # time and iteration count should be rolled back by a not successful call of advance
            desired_output = (self.t_cp_mocked, self.n_cp_mocked, precice_step_complete, self.dt)
            self.assertEqual(precice.advance(None, self.u_np1_mocked, self.u_n_mocked, self.t, self.dt, self.n),
                             desired_output)

            # we expect that self.u_n_mocked.value has been rolled back to self.u_cp_mocked.value
            self.assertEqual(self.u_n_mocked.value, self.u_cp_mocked.value)

            # we expect that precice._u_cp.value has not been updated
            self.assertEqual(precice._u_cp.value, self.u_cp_mocked.value)

    def test_perform_substep(self):
        fake_PySolverInterfaceClass = self.mock_the_class()
        import fenicsadapter
        with patch('PySolverInterface.PySolverInterface') as fake_PySolverInterfaceClass:
            precice = fenicsadapter.Adapter(self.dummy_config_WR)
            self.mock_the_adapter(precice, fake_PySolverInterfaceClass)

            u0 = MagicMock(name="u0")
            u1 = MagicMock(name="u1")
            u1new = MagicMock(name="u1_new")
            v0 = MagicMock(name="v0")
            v1 = MagicMock(name="v1")

            precice._write_data = [u0, u1]
            precice._read_data = [v0, v1]
            precice.convert_fenics_to_precice = MagicMock(return_value=u1new)

            precice._perform_substep(u1new, self.t, self.dt, self.n)

            self.assertEqual(precice._write_data[0], u0)
            self.assertEqual(precice._write_data[1], u1new)
            self.assertEqual(precice._read_data[0], v0)
            self.assertEqual(precice._read_data[1], v1)

    def test_advance_continue(self):
        """
        Test correct checkpointing, if advance did succeed, but we do not write a checkpoint (for example, if we do subcycling)
        :param fake_PySolverInterface_PySolverInterface: mock instance of PySolverInterface.PySolverInterface
        """
        fake_PySolverInterfaceClass = self.mock_the_class()
        import fenicsadapter
        with patch('PySolverInterface.PySolverInterface') as fake_PySolverInterfaceClass:
            precice = fenicsadapter.Adapter(self.dummy_config)
            self.mock_the_adapter(precice, fake_PySolverInterfaceClass)

            precice_step_complete = False
            # time and iteration count should be rolled back by a not successful call of advance
            desired_output = (self.t + self.dt, self.n + 1, precice_step_complete, self.dt)
            self.assertEqual(precice.advance(None, self.u_np1_mocked, self.u_n_mocked, self.t, self.dt, self.n),
                             desired_output)

            # we expect that self.u_n_mocked.value has been updated to self.u_np1_mocked.value
            self.assertEqual(self.u_n_mocked.value, self.u_np1_mocked.value)

            # we expect that precice._u_cp.value has not been updated
            self.assertEqual(precice._u_cp.value, self.u_cp_mocked.value)

@patch.dict('sys.modules', **{'PySolverInterface': fake_PySolverInterface, 'dolfin': fake_dolfin})
class TestIsCouplingOngoing(TestCase):
    dummy_config = "tests/precice-adapter-config.json"

    def setUp(self):
        warnings.simplefilter('ignore', category=ImportWarning)

    def mock_the_adapter(self, precice, mocked_class):
        """
        We partially mock the fenicsadapter, since proper configuration and initialization of the adapter is not
        necessary to test checkpointing.
        :param precice: the fenicsadapter
        """
        # define functions that are called by advance, but not necessary for the test
        precice._interface = mocked_class()
        """
        precice.extract_coupling_boundary_coordinates = MagicMock(return_value=(None, None))
        precice.convert_fenics_to_precice = MagicMock()
        precice._coupling_bc_expression = MagicMock()
        precice._coupling_bc_expression.update_boundary_data = MagicMock()
        # initialize checkpointing manually
        precice._t_cp = self.t_cp_mocked
        precice._u_cp = self.u_cp_mocked
        precice._n_cp = self.n_cp_mocked
        """

    def mock_the_class(self, is_coupling_ongoing_return=False):
        """
        We mock the PySolverInterface such that we can easily get the desired behavior.
        :param is_coupling_ongoing_return: specify whether the interface isCouplingOngoing should return True or False
        """
        mocked_class = Mock
        mocked_class.isCouplingOngoing = MagicMock(return_value=is_coupling_ongoing_return)
        return mocked_class

    def test_isCouplingOngoing(self):
        fake_PySolverInterfaceClass = self.mock_the_class(is_coupling_ongoing_return=True)
        import fenicsadapter
        with patch('PySolverInterface.PySolverInterface') as fake_PySolverInterfaceClass:
            precice = fenicsadapter.Adapter(self.dummy_config)
            self.mock_the_adapter(precice, fake_PySolverInterfaceClass)
            self.assertEqual(precice.is_coupling_ongoing(), True)
            """
            precice._interface.isCouplingOngoing = MagicMock(return_value=False)
            self.assertEqual(precice.is_coupling_ongoing(), False)
            """
