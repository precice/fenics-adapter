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
class MyTest(TestCase):

    dt = 1
    n = 0
    t = 0
    u_n_mocked = MockedArray()
    u_np1_mocked = MockedArray()

    def setUp(self):
        fake_PySolverInterface.PyActionReadIterationCheckpoint = MagicMock(return_value=1)
        fake_PySolverInterface.PyActionWriteIterationCheckpoint = MagicMock(return_value=2)
        warnings.simplefilter('ignore', category=ImportWarning)

    @patch('PySolverInterface.PySolverInterface')
    def test_advance_success(self, fake_PySolverInterface_PySolverInterface):
        import PySolverInterface

        success = True
        desired_out = (self.t + self.dt, self.n + 1, success)

        def mock_feedback(py_action):
            if py_action == PySolverInterface.PyActionReadIterationCheckpoint():
                return False
            elif py_action == PySolverInterface.PyActionWriteIterationCheckpoint():
                return True

        fake_PySolverInterface_PySolverInterface.return_value.isActionRequired = MagicMock(side_effect=mock_feedback)
        fake_PySolverInterface_PySolverInterface.return_value.writeBlockScalarData = MagicMock()
        fake_PySolverInterface_PySolverInterface.return_value.readBlockScalarData = MagicMock()
        fake_PySolverInterface_PySolverInterface.return_value.advance = MagicMock()

        import fenicsadapter
        precice = fenicsadapter.Adapter()
        precice.configure(None, None, None, None, None)
        precice.extract_coupling_boundary_coordinates = MagicMock(return_value=(None, None))
        precice.convert_fenics_to_precice = MagicMock()
        precice._coupling_bc_expression = MagicMock()
        precice._coupling_bc_expression.update_boundary_data = MagicMock()

        precice._t_cp = self.t
        precice._u_cp = self.u_n_mocked
        precice._n_cp = self.n

        value_u_np1 = self.u_np1_mocked.value

        self.assertEqual(precice.advance(None, self.u_np1_mocked, self.u_n_mocked, self.t, self.dt, self.n),
                         desired_out)
        self.assertEqual(self.u_n_mocked.value, value_u_np1)
        self.assertEqual(precice._u_cp.value, value_u_np1)

    @patch('PySolverInterface.PySolverInterface')
    def test_advance_rollback(self, fake_PySolverInterface_PySolverInterface):
        import PySolverInterface

        success = False
        desired_out = (self.t, self.n, success)

        def mock_feedback(py_action):
            if py_action == PySolverInterface.PyActionReadIterationCheckpoint():
                return True
            elif py_action == PySolverInterface.PyActionWriteIterationCheckpoint():
                return False

        fake_PySolverInterface_PySolverInterface.return_value.isActionRequired = MagicMock(side_effect=mock_feedback)
        fake_PySolverInterface_PySolverInterface.return_value.writeBlockScalarData = MagicMock()
        fake_PySolverInterface_PySolverInterface.return_value.readBlockScalarData = MagicMock()
        fake_PySolverInterface_PySolverInterface.return_value.advance = MagicMock()

        import fenicsadapter
        precice = fenicsadapter.Adapter()
        precice.configure(None, None, None, None, None)
        precice.extract_coupling_boundary_coordinates = MagicMock(return_value=(None, None))
        precice.convert_fenics_to_precice = MagicMock()
        precice._coupling_bc_expression = MagicMock()
        precice._coupling_bc_expression.update_boundary_data = MagicMock()

        precice._t_cp = self.t
        precice._u_cp = self.u_n_mocked
        precice._n_cp = self.n

        value_u_n = self.u_n_mocked.value

        self.assertEqual(precice.advance(None, self.u_np1_mocked, self.u_n_mocked, self.t, self.dt, self.n),
                         desired_out)
        self.assertEqual(self.u_n_mocked.value, value_u_n)
        self.assertEqual(precice._u_cp.value, value_u_n)

    @patch('PySolverInterface.PySolverInterface')
    def test_isCouplingOngoing(self, fake_PySolverInterface_PySolverInterface):
        import fenicsadapter
        precice = fenicsadapter.Adapter()
        precice.configure(None, None, None, None, None)

        fake_PySolverInterface_PySolverInterface.return_value.isCouplingOngoing = MagicMock(return_value=True)
        self.assertEqual(precice.is_coupling_ongoing(), True)

        fake_PySolverInterface_PySolverInterface.return_value.isCouplingOngoing = MagicMock(return_value=False)
        self.assertEqual(precice.is_coupling_ongoing(), False)
