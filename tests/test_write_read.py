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
class TestWriteandReadData(TestCase):
    """
    Test suite to test read and write functionality of Adapter. Read and Write functionality is tested for both scalar
    and vector data.
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

    def test_scalar_write(self):
        """
        Test to check if Adapter function write() passes correct parameters to the API function write_block_scalar_data()
        """
        from precice import Interface
        import fenicsadapter

        Interface.write_block_scalar_data = MagicMock()
        Interface.get_dimensions = MagicMock(return_value=2)
        Interface.get_mesh_id = MagicMock()
        Interface.get_data_id = MagicMock(return_value=self.fake_id)

        precice = fenicsadapter.Adapter(self.dummy_config)
        precice._interface = Interface(None, None, None, None)
        precice._coupling_mesh_vertices = np.stack([self.vertices_x, self.vertices_y], axis=1)
        precice._write_data_id = self.fake_id
        precice._vertex_ids = np.arange(self.n_vertices)

        precice.write_data(self.scalar_function)

        expected_data_id = self.fake_id
        expected_values = np.array([self.scalar_expr(x_right, y) for y in self.vertices_y])
        expected_ids = np.arange(self.n_vertices)
        expected_args = [expected_data_id, expected_ids, expected_values]

        for arg, expected_arg in zip(Interface.write_block_scalar_data.call_args[0], expected_args):
            if type(arg) is int:
                self.assertTrue(arg == expected_arg)
            elif type(arg) is np.ndarray:
                np.testing.assert_allclose(arg, expected_arg)

    def test_vector_write(self):
        """
        Test to check if Adapter function write() passes correct parameters to the API function write_block_vector_data()
        """
        from precice import Interface
        import fenicsadapter

        Interface.write_block_vector_data = MagicMock()
        Interface.get_dimensions = MagicMock(return_value=self.dimension)
        Interface.get_mesh_id = MagicMock()
        Interface.get_data_id = MagicMock(return_value=self.fake_id)

        precice = fenicsadapter.Adapter(self.dummy_config)
        precice._interface = Interface(None, None, None, None)
        precice._coupling_mesh_vertices = np.stack([self.vertices_x, self.vertices_y], axis=1)
        precice._write_data_id = self.fake_id
        precice._vertex_ids = np.arange(self.n_vertices)
        precice._fenics_dimensions = self.dimension

        precice.write_data(self.vector_function)

        expected_data_id = self.fake_id
        expected_values_x = np.array([self.vector_expr(x_right, y)[0] for y in np.linspace(y_bottom, y_top, 11)])
        expected_values_y = np.array([self.vector_expr(x_right, y)[1] for y in np.linspace(y_bottom, y_top, 11)])
        expected_values = np.stack([expected_values_x, expected_values_y], axis=1)
        expected_ids = np.arange(self.n_vertices)
        expected_args = [expected_data_id, expected_ids, expected_values]

        for arg, expected_arg in zip(Interface.write_block_vector_data.call_args[0], expected_args):
            if type(arg) is int:
                self.assertTrue(arg == expected_arg)
            elif type(arg) is np.ndarray:
                np.testing.assert_almost_equal(arg, expected_arg)

    def test_scalar_read(self):
        """
        Test to check if Adapter function read() passes correct parameters to the API function read_block_scalar_data()
        Test to check if data return by API function read_block_scalar_data() is also returned by Adapter function read()
        """
        from precice import Interface
        import fenicsadapter
        from fenicsadapter.adapter_core import FunctionType

        def return_dummy_data(n_points):
            data = np.arange(n_points)
            return data

        Interface.read_block_scalar_data = MagicMock(return_value=return_dummy_data(self.n_vertices))
        Interface.get_dimensions = MagicMock(return_value=self.dimension)
        Interface.get_mesh_id = MagicMock()
        Interface.get_data_id = MagicMock(return_value=self.fake_id)

        precice = fenicsadapter.Adapter(self.dummy_config)
        precice._interface = Interface(None, None, None, None)
        precice._read_function_type = FunctionType.SCALAR
        precice._fenics_dimensions = self.dimension
        precice._coupling_mesh_vertices = np.stack([self.vertices_x, self.vertices_y], axis=1)
        precice._read_data_id = self.fake_id
        precice._vertex_ids = np.arange(self.n_vertices)

        read_data = precice.read_data()

        expected_data_id = self.fake_id
        expected_ids = np.arange(self.n_vertices)
        expected_args = [expected_data_id, expected_ids]

        for arg, expected_arg in zip(Interface.read_block_scalar_data.call_args[0], expected_args):
            if type(arg) is int:
                self.assertTrue(arg == expected_arg)
            elif type(arg) is np.ndarray:
                np.testing.assert_allclose(arg, expected_arg)

        np.testing.assert_almost_equal(read_data, return_dummy_data(self.n_vertices))

    def test_vector_read(self):
        """
        Test to check if Adapter function read() passes correct parameters to the API function read_block_vector_data()
        Test to check if data return by API function read_block_vector_data() is also returned by Adapter function read()
        """
        from precice import Interface
        import fenicsadapter
        from fenicsadapter.adapter_core import FunctionType

        def return_dummy_data(n_points):
            data = np.arange(n_points * self.dimension).reshape(n_points, self.dimension)
            return data

        Interface.read_block_vector_data = MagicMock(return_value=return_dummy_data(self.n_vertices))
        Interface.get_dimensions = MagicMock(return_value=self.dimension)
        Interface.get_mesh_id = MagicMock()
        Interface.get_data_id = MagicMock(return_value=self.fake_id)

        precice = fenicsadapter.Adapter(self.dummy_config)
        precice._interface = Interface(None, None, None, None)
        precice._read_function_type = FunctionType.VECTOR
        precice._coupling_mesh_vertices = np.stack([self.vertices_x, self.vertices_y], axis=1)
        precice._read_data_id = self.fake_id
        precice._vertex_ids = np.arange(self.n_vertices)
        precice._fenics_dimensions = self.dimension

        read_data = precice.read_data()

        expected_data_id = self.fake_id
        expected_ids = np.arange(self.n_vertices)
        expected_args = [expected_data_id, expected_ids]

        for arg, expected_arg in zip(Interface.read_block_vector_data.call_args[0], expected_args):
            if type(arg) is int:
                self.assertTrue(arg == expected_arg)
            elif type(arg) is np.ndarray:
                np.testing.assert_allclose(arg, expected_arg)

        np.testing.assert_almost_equal(read_data, return_dummy_data(self.n_vertices))
