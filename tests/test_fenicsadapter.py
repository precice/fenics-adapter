# comments on test layout: https://docs.pytest.org/en/latest/goodpractices.html
# run with python -m unittest tests.test_fenicsadapter

from unittest.mock import MagicMock, patch
from unittest import TestCase
import warnings
import tests.MockedPrecice

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
    write_function_mocked = MockedArray()
    u_cp_mocked = MockedArray()  # value of the checkpoint
    t_cp_mocked = t  # time for the checkpoint
    n_cp_mocked = n  # iteration count for the checkpoint
    dummy_config = "tests/precice-adapter-config.json"

    # todo if we support multirate, we should use the lines below for checkpointing
    # for the general case the checkpoint u_cp (and t_cp and n_cp) can differ from u_n and u_np1
    # t_cp_mocked = MagicMock()  # time for the checkpoint
    # n_cp_mocked = nMagicMock()  # iteration count for the checkpoint

    def test_checkpoint_mechanism(self):
        """
        Test correct checkpoint storing
        """
        import fenicsadapter
        from precice import Interface, action_write_iteration_checkpoint

        def is_action_required_behavior(py_action):
            if py_action == action_write_iteration_checkpoint():
                return True
            else:
                return False

        Interface.initialize = MagicMock(return_value=self.dt)
        Interface.is_action_required = MagicMock(side_effect=is_action_required_behavior)
        Interface.get_dimensions = MagicMock()
        Interface.get_mesh_id = MagicMock()
        Interface.get_data_id = MagicMock()
        Interface.mark_action_fulfilled = MagicMock()
        Interface.is_time_window_complete = MagicMock(return_value=True)
        Interface.advance = MagicMock()

        precice = fenicsadapter.Adapter(self.dummy_config)

        precice.store_checkpoint(self.u_n_mocked, self.t, self.n)

        # Replicating valid control work flow
        precice.advance(self.dt)
        Interface.is_time_window_complete = MagicMock(return_value=False)

        # Check if the checkpoint is stored correctly in the adapter
        self.assertEqual(precice.retrieve_checkpoint() == self.u_n_mocked, self.t, self.n)


@patch.dict('sys.modules', **{'dolfin': fake_dolfin, 'precice': tests.MockedPrecice})
class TestIsCouplingOngoing(TestCase):
    dummy_config = "tests/precice-adapter-config.json"

    def test_isCouplingOngoing(self):
        """
        A unit test to check if the isCouplingOngoing boolean is correctly communicated to the Interface
        :return:
        """
        import fenicsadapter
        from precice import Interface

        Interface.is_coupling_ongoing = MagicMock(return_value=True)
        Interface.get_dimensions = MagicMock(return_value=2)
        Interface.get_mesh_id = MagicMock()
        Interface.get_data_id = MagicMock()

        precice = fenicsadapter.Adapter(self.dummy_config)

        self.assertEqual(precice.is_coupling_ongoing(), True)
