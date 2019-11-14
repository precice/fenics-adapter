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

    mesh = UnitSquareMesh(10, 10)

    scalar_expr = Expression("x[0]*x[0] + x[1]*x[1]", degree=2)
    scalar_V = FunctionSpace(mesh, "P", 2)
    scalar_function = interpolate(scalar_expr, scalar_V)

    vector_expr = Expression(("x[0] + x[1]*x[1]", "x[0] - x[1]*x[1]"), degree=2)
    vector_V = VectorFunctionSpace(mesh, "P", 2)
    vector_function = interpolate(vector_expr, vector_V)

    def setUp(self):
        pass

    def test_write_scalar_data(self):
        from precice import Interface
        import fenicsadapter

        def return_dummy_vertices(unused1, unused2, unused3, vertices):
            for i in range(len(vertices)):
                vertices[i] = i

        Interface.configure = MagicMock()
        Interface.write_block_scalar_data = MagicMock()
        Interface.read_block_vector_data = MagicMock()
        Interface.get_dimensions = MagicMock(return_value=2)
        Interface.set_mesh_vertices = MagicMock(return_value=return_dummy_vertices)
        Interface.initialize = MagicMock()
        Interface.initialize_data = MagicMock()
        Interface.is_action_required = MagicMock(return_value=False)
        Interface.fulfilled_action = MagicMock()
        Interface.advance = MagicMock()
        Interface.get_mesh_id = MagicMock()
        Interface.get_data_id = MagicMock(return_value=15)
        Interface.is_read_data_available = MagicMock(return_value=False)
        Interface.set_mesh_edge = MagicMock()

        write_u = self.scalar_function
        read_u = self.vector_function
        u_init = self.scalar_function

        precice = fenicsadapter.Adapter(self.dummy_config)
        precice._coupling_bc_expression = MagicMock()
        precice.initialize(RightBoundary(), self.mesh, read_u, write_u, u_init)
        precice.advance(write_u, u_init, u_init, 0, 0, 0)

        expected_data_id = 15
        expected_size = 11
        expected_values = np.array([self.scalar_expr(x_right, y) for y in np.linspace(y_bottom, y_top, 11)])
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

        def return_dummy_vertices(unused1, unused2, unused3, vertices):
            for i in range(len(vertices)):
                vertices[i] = i

        Interface.configure = MagicMock()
        Interface.write_block_vector_data = MagicMock()
        Interface.read_block_scalar_data = MagicMock()
        Interface.get_dimensions = MagicMock(return_value=2)
        Interface.set_mesh_vertices = MagicMock(return_value=return_dummy_vertices)
        Interface.initialize = MagicMock()
        Interface.initialize_data = MagicMock()
        Interface.is_action_required = MagicMock(return_value=False)
        Interface.fulfilled_action = MagicMock()
        Interface.advance = MagicMock()
        Interface.get_mesh_id = MagicMock()
        Interface.get_data_id = MagicMock(return_value=15)
        Interface.is_read_data_available = MagicMock(return_value=False)
        Interface.set_mesh_edge = MagicMock()

        write_u = self.vector_function
        read_u = self.scalar_function
        u_init = self.vector_function

        precice = fenicsadapter.Adapter(self.dummy_config)
        precice._coupling_bc_expression = MagicMock()
        precice.initialize(RightBoundary(), self.mesh, read_u, write_u, u_init)
        precice.advance(write_u, u_init, u_init, 0, 0, 0)

        expected_data_id = 15
        expected_size = 11
        expected_values_x = np.array([self.vector_expr(x_right, y)[0] for y in np.linspace(y_bottom, y_top, 11)])
        expected_values_y = np.array([self.vector_expr(x_right, y)[1] for y in np.linspace(y_bottom, y_top, 11)])
        expected_values = np.stack([expected_values_x, expected_values_y]).T.ravel()
        expected_ids = np.zeros(11)

        expected_args = [expected_data_id, expected_size, expected_ids, expected_values]

        for arg, expected_arg in zip(Interface.write_block_vector_data.call_args[0], expected_args):
            if type(arg) is int:
                self.assertTrue(arg == expected_arg)
            elif type(arg) is np.ndarray:
                np.testing.assert_almost_equal(arg, expected_arg)

    def test_read_scalar_data(self):
        from precice import Interface
        import fenicsadapter

        def return_dummy_data(unused1, unused2, unused3, read_data):
            for i in range(len(read_data)):
                read_data[i] = i

        def return_dummy_vertices(unused1, unused2, unused3, vertices):
            for i in range(len(vertices)):
                vertices[i] = i

        Interface.configure = MagicMock()
        Interface.write_block_vector_data = MagicMock()
        Interface.read_block_scalar_data = MagicMock(side_effect=return_dummy_data)
        Interface.get_dimensions = MagicMock(return_value=2)
        Interface.set_mesh_vertices = MagicMock(return_value=return_dummy_vertices)
        Interface.initialize = MagicMock()
        Interface.initialize_data = MagicMock()
        Interface.is_action_required = MagicMock(return_value=False)
        Interface.fulfilled_action = MagicMock()
        Interface.advance = MagicMock()
        Interface.get_mesh_id = MagicMock()
        Interface.get_data_id = MagicMock(return_value=15)
        Interface.is_read_data_available = MagicMock(return_value=False)
        Interface.set_mesh_edge = MagicMock()

        write_u = self.vector_function
        read_u = self.scalar_function
        u_init = self.vector_function

        precice = fenicsadapter.Adapter(self.dummy_config)
        precice._coupling_bc_expression = MagicMock()
        precice.initialize(RightBoundary(), self.mesh, read_u, write_u, u_init)
        precice.advance(write_u, u_init, u_init, 0, 0, 0)

        expected_data_id = 15
        expected_size = 11
        expected_values = np.array([i for i in range(11)])
        expected_ids = np.zeros(11)

        expected_args = [expected_data_id, expected_size, expected_ids, expected_values]

        for arg, expected_arg in zip(Interface.read_block_scalar_data.call_args[0], expected_args):
            if type(arg) is int:
                self.assertTrue(arg == expected_arg)
            elif type(arg) is np.ndarray:
                np.testing.assert_allclose(arg, expected_arg)

    def test_read_vector_data(self):
        from precice import Interface
        import fenicsadapter

        def return_dummy_data(unused1, unused2, unused3, read_data):
            for i in range(len(read_data)):
                read_data[i] = i

        def return_dummy_vertices(unused1, unused2, unused3, vertices):
            for i in range(len(vertices)):
                vertices[i] = i

        Interface.configure = MagicMock()
        Interface.write_block_scalar_data = MagicMock()
        Interface.read_block_vector_data = MagicMock(side_effect=return_dummy_data)
        Interface.get_dimensions = MagicMock(return_value=2)
        Interface.set_mesh_vertices = MagicMock(return_value=return_dummy_vertices)
        Interface.initialize = MagicMock()
        Interface.initialize_data = MagicMock()
        Interface.is_action_required = MagicMock(return_value=False)
        Interface.fulfilled_action = MagicMock()
        Interface.advance = MagicMock()
        Interface.get_mesh_id = MagicMock()
        Interface.get_data_id = MagicMock(return_value=15)
        Interface.is_read_data_available = MagicMock(return_value=False)
        Interface.set_mesh_edge = MagicMock()

        write_u = self.scalar_function
        read_u = self.vector_function
        u_init = self.scalar_function

        precice = fenicsadapter.Adapter(self.dummy_config)
        precice._coupling_bc_expression = MagicMock()
        precice.initialize(RightBoundary(), self.mesh, read_u, write_u, u_init)
        precice.advance(write_u, u_init, u_init, 0, 0, 0)

        expected_data_id = 15
        expected_size = 11
        expected_values = np.array([i for i in range(2 * 11)])
        expected_ids = np.zeros(11)

        expected_args = [expected_data_id, expected_size, expected_ids, expected_values]

        for arg, expected_arg in zip(Interface.read_block_vector_data.call_args[0], expected_args):
            if type(arg) is int:
                self.assertTrue(arg == expected_arg)
            elif type(arg) is np.ndarray:
                np.testing.assert_allclose(arg, expected_arg)
