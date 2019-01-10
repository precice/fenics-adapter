# comments on test layout: https://docs.pytest.org/en/latest/goodpractices.html
# run with python -m unittest tests.test_fenicsadapter

from unittest.mock import MagicMock, patch
from unittest import TestCase
import warnings

fake_dolfin = MagicMock()
fake_PySolverInterface = MagicMock()


@patch.dict('sys.modules', **{'PySolverInterface': fake_PySolverInterface, 'dolfin': fake_dolfin})
class MyTest(TestCase):

    dt = 1

    n = 0
    t_n = 0
    u_n_mocked = MagicMock()
    u_n_mocked.copy.return_value = u_n_mocked

    np1 = 1
    u_np1_mocked = MagicMock()
    u_np1_mocked.copy.return_value = u_np1_mocked
    t_np1 = t_n + dt

    def setUp(self):
        fake_PySolverInterface.PyActionReadIterationCheckpoint = MagicMock(return_value=1)
        fake_PySolverInterface.PyActionWriteIterationCheckpoint = MagicMock(return_value=2)
        warnings.simplefilter('ignore', category=ImportWarning)

    @patch('PySolverInterface.PySolverInterface')
    def test_advance_success(self, fake_PySolverInterface_PySolverInterface):
        import PySolverInterface

        readIterationCheckpointOut = False
        success = not readIterationCheckpointOut
        desiredOutputOfAdvance = (self.u_np1_mocked, self.t_np1, self.np1, success)

        def mock_feedback(input):
            if input == PySolverInterface.PyActionReadIterationCheckpoint():
                return readIterationCheckpointOut
            elif input == PySolverInterface.PyActionWriteIterationCheckpoint():
                return not readIterationCheckpointOut

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

        precice._t_cp = self.t_n
        precice._u_cp = self.u_n_mocked
        precice._n_cp = self.n

        self.assertEqual(precice.advance(None, self.u_np1_mocked, self.t_np1, self.np1, self.dt),
                         desiredOutputOfAdvance)

    @patch('PySolverInterface.PySolverInterface')
    def test_advance_rollback(self, fake_PySolverInterface_PySolverInterface):
        import PySolverInterface

        readIterationCheckpointOut = True
        success = not readIterationCheckpointOut
        desiredOutputOfAdvance = (self.u_n_mocked, self.t_n, self.n, success)

        def mock_feedback(input):
            if input == PySolverInterface.PyActionReadIterationCheckpoint():
                return readIterationCheckpointOut
            elif input == PySolverInterface.PyActionWriteIterationCheckpoint():
                return not readIterationCheckpointOut

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

        precice._t_cp = self.t_n
        precice._u_cp = self.u_n_mocked
        precice._n_cp = self.n

        self.assertEqual(precice.advance(None, self.u_np1_mocked, self.t_np1, self.np1, self.dt),
                         desiredOutputOfAdvance)

    @patch('PySolverInterface.PySolverInterface')
    def test_isCouplingOngoing(self, fake_PySolverInterface_PySolverInterface):
        import fenicsadapter
        precice = fenicsadapter.Adapter()
        precice.configure(None, None, None, None, None)

        fake_PySolverInterface_PySolverInterface.return_value.isCouplingOngoing = MagicMock(return_value=True)
        self.assertEqual(precice.is_coupling_ongoing(), True)

        fake_PySolverInterface_PySolverInterface.return_value.isCouplingOngoing = MagicMock(return_value=False)
        self.assertEqual(precice.is_coupling_ongoing(), False)
