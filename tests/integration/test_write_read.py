from unittest.mock import MagicMock, patch
from unittest import TestCase
from tests import MockedPrecice
from fenics import Expression, UnitSquareMesh, FunctionSpace, VectorFunctionSpace, interpolate, SubDomain, near
import numpy as np

x_left, x_right = 0, 1
y_bottom, y_top = 0, 1

dummy_dt = 1


class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14
        if on_boundary and near(x[0], x_right, tol):
            return True
        else:
            return False


@patch.dict('sys.modules', {'precice': MockedPrecice})
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
    vertices_x = [x_right for _ in range(n_vertices)]
    vertices_y = np.linspace(y_bottom, y_top, n_vertices)

    def test_scalar_write(self):
        """
        Test to check if Adapter function write() passes correct parameters to the API function write_block_scalar_data()
        """
        from precice import Participant
        import fenicsprecice

        Participant.write_data = MagicMock()
        Participant.get_mesh_dimensions = MagicMock(return_value=2)
        Participant.set_mesh_vertices = MagicMock(return_value=np.arange(self.n_vertices))
        Participant.requires_mesh_connectivity_for = MagicMock(return_value=False)
        Participant.requires_initial_data = MagicMock(return_value=False)
        Participant.initialize = MagicMock()

        precice = fenicsprecice.Adapter(self.dummy_config)
        precice._participant = Participant(None, None, None, None)
        precice.initialize(RightBoundary(), self.scalar_V, self.scalar_function)

        precice.write_data(self.scalar_function)

        expected_values = np.array([self.scalar_expr(x_right, y) for y in self.vertices_y])
        expected_ids = np.arange(self.n_vertices)
        expected_args = [expected_ids, expected_values]

        for arg, expected_arg in zip(Participant.write_data.call_args[0], expected_args):
            if isinstance(arg, int):
                self.assertTrue(arg == expected_arg)
            elif isinstance(arg, np.ndarray):
                np.testing.assert_allclose(arg, expected_arg)

    def test_vector_write(self):
        """
        Test to check if Adapter function write() passes correct parameters to the API function write_block_vector_data()
        """
        from precice import Participant
        import fenicsprecice

        Participant.write_data = MagicMock()
        Participant.get_mesh_dimensions = MagicMock(return_value=self.dimension)
        Participant.set_mesh_vertices = MagicMock(return_value=np.arange(self.n_vertices))
        Participant.requires_mesh_connectivity_for = MagicMock(return_value=False)
        Participant.requires_initial_data = MagicMock(return_value=False)
        Participant.initialize = MagicMock()

        precice = fenicsprecice.Adapter(self.dummy_config)
        precice._participant = Participant(None, None, None, None)
        precice.initialize(RightBoundary(), self.vector_V, self.vector_function)

        precice.write_data(self.vector_function)

        expected_values_x = np.array([self.vector_expr(x_right, y)[0] for y in np.linspace(y_bottom, y_top, 11)])
        expected_values_y = np.array([self.vector_expr(x_right, y)[1] for y in np.linspace(y_bottom, y_top, 11)])
        expected_values = np.stack([expected_values_x, expected_values_y], axis=1)
        expected_ids = np.arange(self.n_vertices)
        expected_args = [expected_ids, expected_values]

        for arg, expected_arg in zip(Participant.write_data.call_args[0], expected_args):
            if isinstance(arg, int):
                self.assertTrue(arg == expected_arg)
            elif isinstance(arg, np.ndarray):
                print(arg)
                print(expected_arg)
                np.testing.assert_almost_equal(arg, expected_arg)

    def test_scalar_read(self):
        """
        Test to check if Adapter function read() passes correct parameters to the API function read_block_scalar_data()
        Test to check if data return by API function read_block_scalar_data() is also returned by Adapter function read()
        """
        from precice import Participant
        import fenicsprecice

        def return_dummy_data(n_points):
            data = np.arange(n_points)
            return data

        Participant.read_data = MagicMock(return_value=return_dummy_data(self.n_vertices))
        Participant.get_mesh_dimensions = MagicMock(return_value=self.dimension)
        Participant.set_mesh_vertices = MagicMock(return_value=np.arange(self.n_vertices))
        Participant.requires_mesh_connectivity_for = MagicMock(return_value=False)
        Participant.requires_initial_data = MagicMock(return_value=False)
        Participant.initialize = MagicMock()
        Participant.get_max_time_step_size = MagicMock(return_value=dummy_dt)

        precice = fenicsprecice.Adapter(self.dummy_config)
        precice._participant = Participant(None, None, None, None)
        precice.initialize(RightBoundary(), self.scalar_V)

        dt = precice.get_max_time_step_size()
        read_data = precice.read_data(dt)

        expected_ids = np.arange(self.n_vertices)
        expected_args = ["Dummy-Mesh", "Dummy-Read", expected_ids, dummy_dt]

        for arg, expected_arg in zip(Participant.read_data.call_args[0], expected_args):
            if isinstance(arg, int) or isinstance(arg, str):
                self.assertTrue(arg == expected_arg)
            elif isinstance(arg, np.ndarray):
                np.testing.assert_allclose(arg, expected_arg)
            else:
                self.fail(f"Unexpected combination of arg: {arg}, expected_arg: {expected_arg}")

        np.testing.assert_almost_equal(list(read_data.values()), return_dummy_data(self.n_vertices))

    def test_vector_read(self):
        """
        Test to check if Adapter function read() passes correct parameters to the API function read_block_vector_data()
        Test to check if data return by API function read_block_vector_data() is also returned by Adapter function read()
        """
        from precice import Participant
        import fenicsprecice

        def return_dummy_data(n_points):
            data = np.arange(n_points * self.dimension).reshape(n_points, self.dimension)
            return data

        Participant.read_data = MagicMock(return_value=return_dummy_data(self.n_vertices))
        Participant.get_mesh_dimensions = MagicMock(return_value=self.dimension)
        Participant.set_mesh_vertices = MagicMock(return_value=np.arange(self.n_vertices))
        Participant.requires_mesh_connectivity_for = MagicMock(return_value=False)
        Participant.requires_initial_data = MagicMock(return_value=False)
        Participant.initialize = MagicMock()
        Participant.get_max_time_step_size = MagicMock(return_value=dummy_dt)

        precice = fenicsprecice.Adapter(self.dummy_config)
        precice._participant = Participant(None, None, None, None)
        precice.initialize(RightBoundary(), self.vector_V)

        dt = precice.get_max_time_step_size()
        read_data = precice.read_data(dt)

        expected_ids = np.arange(self.n_vertices)
        expected_args = ["Dummy-Mesh", "Dummy-Read", expected_ids, dummy_dt]

        for arg, expected_arg in zip(Participant.read_data.call_args[0], expected_args):
            if isinstance(arg, int) or isinstance(arg, str):
                self.assertTrue(arg == expected_arg)
            elif isinstance(arg, np.ndarray):
                np.testing.assert_allclose(arg, expected_arg)
            else:
                self.fail(f"Unexpected combination of arg: {arg}, expected_arg: {expected_arg}")

        np.testing.assert_almost_equal(list(read_data.values()), return_dummy_data(self.n_vertices))
