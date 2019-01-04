# comments on test layout: https://docs.pytest.org/en/latest/goodpractices.html
# run with python -m unittest tests.test_fenicsadapter

from unittest.mock import MagicMock, patch
from unittest import TestCase

fake_dolfin = MagicMock()
fake_PySolverInterface = MagicMock()


@patch.dict('sys.modules', **{'PySolverInterface': fake_PySolverInterface, 'dolfin': fake_dolfin})
class MyTest(TestCase):
    @patch('PySolverInterface.PySolverInterface')
    def test_advance_success(self, fake_PySolverInterface_PySolverInterface):
        import dolfin
        import PySolverInterface

        assert(dolfin == fake_dolfin)
        assert(PySolverInterface == fake_PySolverInterface)
        import fenicsadapter
        PySolverInterface.PyActionReadIterationCheckpoint = MagicMock(return_value=1)
        PySolverInterface.PyActionWriteIterationCheckpoint = MagicMock(return_value=2)

        readIterationCheckpointOut = False
        desiredOutputOfAdvance = not readIterationCheckpointOut

        def mock_feedback(input):
            if input == PySolverInterface.PyActionReadIterationCheckpoint():
                return readIterationCheckpointOut
            elif input == PySolverInterface.PyActionWriteIterationCheckpoint():
                return (not readIterationCheckpointOut)

        fake_PySolverInterface_PySolverInterface.return_value.isActionRequired = MagicMock(side_effect=mock_feedback)
        fake_PySolverInterface_PySolverInterface.return_value.writeBlockScalarData = MagicMock()
        fake_PySolverInterface_PySolverInterface.return_value.readBlockScalarData = MagicMock()
        fake_PySolverInterface_PySolverInterface.return_value.advance = MagicMock()

        import fenicsadapter as fenicsadapter
        precice = fenicsadapter.Adapter()
        precice.configure(None, None, None, None, None)
        precice.extract_coupling_boundary_coordinates = MagicMock(return_value=(None, None))
        precice.convert_fenics_to_precice = MagicMock()
        precice._coupling_bc_expression = MagicMock()
        precice._coupling_bc_expression.update_boundary_data = MagicMock()

        self.assertEqual(precice.advance(None,0), desiredOutputOfAdvance)
        return 0

    @patch('PySolverInterface.PySolverInterface')
    def test_advance_rollback(self,fake_PySolverInterface_PySolverInterface):
        import dolfin
        import PySolverInterface

        assert(dolfin == fake_dolfin)
        assert(PySolverInterface == fake_PySolverInterface)
        import fenicsadapter
        PySolverInterface.PyActionReadIterationCheckpoint = MagicMock(return_value=1)
        PySolverInterface.PyActionWriteIterationCheckpoint = MagicMock(return_value=2)

        readIterationCheckpointOut = True
        desiredOutputOfAdvance = not readIterationCheckpointOut

        def mock_feedback(input):
            if input == PySolverInterface.PyActionReadIterationCheckpoint():
                return readIterationCheckpointOut
            elif input == PySolverInterface.PyActionWriteIterationCheckpoint():
                return (not readIterationCheckpointOut)

        fake_PySolverInterface_PySolverInterface.return_value.isActionRequired = MagicMock(side_effect=mock_feedback)
        fake_PySolverInterface_PySolverInterface.return_value.writeBlockScalarData = MagicMock()
        fake_PySolverInterface_PySolverInterface.return_value.readBlockScalarData = MagicMock()
        fake_PySolverInterface_PySolverInterface.return_value.advance = MagicMock()

        import fenicsadapter as fenicsadapter
        precice = fenicsadapter.Adapter()
        precice.configure(None, None, None, None, None)
        precice.extract_coupling_boundary_coordinates = MagicMock(return_value=(None, None))
        precice.convert_fenics_to_precice = MagicMock()
        precice._coupling_bc_expression = MagicMock()
        precice._coupling_bc_expression.update_boundary_data = MagicMock()

        self.assertEqual(precice.advance(None,0), desiredOutputOfAdvance)
