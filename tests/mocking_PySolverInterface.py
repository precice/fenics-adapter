from unittest.mock import MagicMock, Mock, patch

PySolverInterface_mock = MagicMock()
dolfin_mock = MagicMock()
with patch.dict('sys.modules', **{
            'PySolverInterface': PySolverInterface_mock,
            'dolfin': dolfin_mock,
        }):

    @patch('PySolverInterface.PySolverInterface')
    def test_things(PySolverInterface_PySolverInterface_mock):
        PySolverInterface_mock.PyActionReadIterationCheckpoint = MagicMock(return_value=1)
        PySolverInterface_mock.PyActionWriteIterationCheckpoint = MagicMock(return_value=2)

        readIterationCheckpointOut = False
        desiredOutputOfAdvance = not readIterationCheckpointOut

        def mock_feedback(input):
            if input == PySolverInterface_mock.PyActionReadIterationCheckpoint():
                return readIterationCheckpointOut
            elif input == PySolverInterface_mock.PyActionWriteIterationCheckpoint():
                return (not readIterationCheckpointOut)

        PySolverInterface_PySolverInterface_mock.return_value.isActionRequired = MagicMock(side_effect=mock_feedback)
        PySolverInterface_PySolverInterface_mock.return_value.writeBlockScalarData = MagicMock()
        PySolverInterface_PySolverInterface_mock.return_value.readBlockScalarData = MagicMock()
        PySolverInterface_PySolverInterface_mock.return_value.advance = MagicMock()

        import fenicsadapter
        precice = fenicsadapter.Adapter()
        precice.configure("a","xxx",1,1,1)
        precice.extract_coupling_boundary_coordinates = MagicMock(return_value=(None, None))
        precice.convert_fenics_to_precice = MagicMock()
        precice._coupling_bc_expression = MagicMock()
        precice._coupling_bc_expression.update_boundary_data = MagicMock()

        print(precice._solver_name)
        print(precice._interface)
        print(PySolverInterface_PySolverInterface_mock.PyActionReadIterationCheckpoint())
        print(PySolverInterface_PySolverInterface_mock.PyActionWriteIterationCheckpoint())
        print(precice._interface.isActionRequired(PySolverInterface_mock.PyActionReadIterationCheckpoint()))
        print(precice._interface.isActionRequired(PySolverInterface_mock.PyActionWriteIterationCheckpoint()))

        assert(precice.advance(None,0) == desiredOutputOfAdvance)

    test_things()
