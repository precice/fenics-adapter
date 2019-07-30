from unittest.mock import MagicMock, patch
from unittest import TestCase
import tests.MockedPrecice
from fenics import Expression, UnitSquareMesh, FunctionSpace, VectorFunctionSpace, interpolate, SubDomain, near
import numpy as np


x_left, x_right = 0, 1
y_bottom, y_top = 0, 1


class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14
        if on_boundary and near(x[0], x_right, tol):
            return True
        else:
            return False


@patch.dict('sys.modules', **{'precice': tests.MockedPrecice})
class TestWriteData(TestCase):
    dummy_config = "tests/precice-adapter-config.json"

    def setUp(self):
        pass

    def test_write_scalar_data(self):
        from precice import Interface
        import fenicsadapter
        Interface.configure = MagicMock()
        Interface.write_block_scalar_data = MagicMock()
        Interface.read_block_scalar_data = MagicMock()
        Interface.get_dimensions = MagicMock(return_value=2)
        Interface.set_mesh_vertices = MagicMock()
        Interface.initialize = MagicMock()
        Interface.initialize_data = MagicMock()
        Interface.is_action_required = MagicMock(return_value=False)
        Interface.fulfilled_action = MagicMock()
        Interface.advance = MagicMock()
        Interface.get_mesh_id = MagicMock()
        Interface.get_data_id = MagicMock(return_value=15)
        Interface.is_read_data_available = MagicMock(return_value=False)

        expr = Expression("x[0]*x[0] + x[1]*x[1]", degree=2)
        mesh = UnitSquareMesh(10, 10)
        V = FunctionSpace(mesh, "P", 2)
        u = interpolate(expr, V)

        precice = fenicsadapter.Adapter(self.dummy_config)
        precice._coupling_bc_expression = MagicMock()
        precice.initialize(RightBoundary(), mesh, u, u, u)
        precice.advance(u, u, u, 0, 0, 0)

        expected_data_id = 15
        expected_size = 11
        expected_values = np.array([x_right**2 + y**2 for y in np.linspace(y_bottom, y_top, 11)])
        expected_ids = np.zeros(11)

        expected_args = [expected_data_id, expected_size, expected_ids, expected_values]

        for arg, expected_arg in zip(Interface.write_block_scalar_data.call_args[0], expected_args):
            if type(arg) is int:
                self.assertTrue(arg == expected_arg)
            elif type(arg) is np.ndarray:
                np.testing.assert_allclose(arg, expected_arg)

    def test_write_vector_data(self):
        from precice import Interface
        import fenicsadapter
        Interface.configure = MagicMock()
        Interface.write_block_vector_data = MagicMock()
        Interface.read_block_vector_data = MagicMock()
        Interface.get_dimensions = MagicMock(return_value=2)
        Interface.set_mesh_vertices = MagicMock()
        Interface.initialize = MagicMock()
        Interface.initialize_data = MagicMock()
        Interface.is_action_required = MagicMock(return_value=False)
        Interface.fulfilled_action = MagicMock()
        Interface.advance = MagicMock()
        Interface.get_mesh_id = MagicMock()
        Interface.get_data_id = MagicMock(return_value=15)
        Interface.is_read_data_available = MagicMock(return_value=False)

        expr = Expression(("x[0]*x[0] + x[1]*x[1]", "x[0]*x[0] - x[1]*x[1]"), degree=2)
        mesh = UnitSquareMesh(10, 10)
        V = VectorFunctionSpace(mesh, "P", 2)
        u = interpolate(expr, V)

        precice = fenicsadapter.Adapter(self.dummy_config)
        precice._coupling_bc_expression = MagicMock()
        precice.initialize(RightBoundary(), mesh, u, u, u)
        precice.advance(u, u, u, 0, 0, 0)

        expected_data_id = 15
        expected_size = 11
        expected_values_x = np.array([x_right ** 2 + y ** 2 for y in np.linspace(y_bottom, y_top, 11)])
        expected_values_y = np.array([x_right ** 2 - y ** 2 for y in np.linspace(y_bottom, y_top, 11)])
        expected_values = np.stack([expected_values_x, expected_values_y]).T.ravel()
        expected_ids = np.zeros(11)

        expected_args = [expected_data_id, expected_size, expected_ids, expected_values]

        for arg, expected_arg in zip(Interface.write_block_vector_data.call_args[0], expected_args):
            if type(arg) is int:
                self.assertTrue(arg == expected_arg)
            elif type(arg) is np.ndarray:
                np.testing.assert_almost_equal(arg, expected_arg)
