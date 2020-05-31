# comments on test layout: https://docs.pytest.org/en/latest/goodpractices.html

from unittest.mock import MagicMock, patch
from unittest import TestCase
import tests.MockedPrecice
import numpy as np
from fenics import Expression, UnitSquareMesh, FunctionSpace, VectorFunctionSpace, interpolate, inner, assemble, dx,\
    project, sqrt

fake_dolfin = MagicMock()
x_left, x_right = 0, 1
y_bottom, y_top = 0, 1


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
    Test suite to check if Checkpointing functionality of the Adapter is working.
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

        # Replicating control flow where implicit iteration has not converged and solver state needs to be restored
        # to a checkpoint
        precice.advance(self.dt)
        Interface.is_time_window_complete = MagicMock(return_value=False)

        # Check if the checkpoint is stored correctly in the adapter
        self.assertEqual(precice.retrieve_checkpoint() == self.u_n_mocked, self.t, self.n)


@patch.dict('sys.modules', **{'dolfin': fake_dolfin, 'precice': tests.MockedPrecice})
class TestExpressionHandling(TestCase):
    """
    Test Expression creating mechanism based on data provided by user.
    """
    dummy_config = "tests/precice-adapter-config.json"

    mesh = UnitSquareMesh(10, 10)
    dimension = 2

    scalar_expr = Expression("x[0]*x[0] + x[1]*x[1]", degree=2)
    scalar_V = FunctionSpace(mesh, "P", 2)
    scalar_function = interpolate(scalar_expr, scalar_V)

    vector_expr = Expression(("x[0] + x[1]*x[1]", "x[0] - x[1]*x[1]"), degree=2)
    vector_V = VectorFunctionSpace(mesh, "P", 2)
    vector_function = interpolate(vector_expr, vector_V)

    n_vertices = 11
    fake_id = 15
    vertices_x = [x_right for _ in range(n_vertices)]
    vertices_y = np.linspace(y_bottom, y_top, n_vertices)

    def test_update_expression(self):
        """
        Check analytical solution with evaluation of coupling expression on the same points
        """
        from precice import Interface
        import fenicsadapter
        from fenicsadapter.expression_core import ExactInterpolationExpression, GeneralInterpolationExpression

        dummy_scalar_data = np.arange(self.n_vertices)

        precice = fenicsadapter.Adapter(self.dummy_config)
        precice._interface = Interface(None, None, None, None)
        precice._coupling_mesh_vertices = np.stack([self.vertices_x, self.vertices_y], axis=1)
        precice._function_space = self.scalar_V
        precice._my_expression = ExactInterpolationExpression

        # Still outputs a mocked form of precice._my_expression in spite of manual initialization above
        print("my_expression initialized as: {}".format(precice._my_expression))

        scalar_coupling_expr = precice.create_coupling_expression(dummy_scalar_data)

        error_normalized = (self.scalar_function - scalar_coupling_expr) / self.scalar_function
        error_pointwise = project(abs(error_normalized), self.scalar_V)
        error_total = sqrt(assemble(inner(error_pointwise, error_pointwise) * dx))

        assert (error_total < 10 ** -4)

        precice._function_space = self.vector_V
        dummy_vector_data = np.arange(self.n_vertices * self.dimension).reshape(self.n_vertices, self.dimension)
        vector_coupling_expr = precice.create_coupling_expression(dummy_vector_data)

        error_normalized = (self.vector_function - vector_coupling_expr) / self.vector_function
        error_pointwise = project(abs(error_normalized), self.vector_V)
        error_total = sqrt(assemble(inner(error_pointwise, error_pointwise) * dx))

        assert (error_total < 10 ** 4)
