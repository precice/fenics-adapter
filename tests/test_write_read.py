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


@patch.dict('sys.modules', **{'precice_future': tests.MockedPrecice})
class TestWriteData(TestCase):
    dummy_config = "tests/precice-adapter-config-WR1.json"
    n_vertices = 11
    dimensions = 2
    n_cells = n_vertices - 1
    mesh = UnitSquareMesh(n_cells, n_cells)
    dummy_vertex_ids = np.arange(n_vertices)

    scalar_expr = Expression("x[0]*x[0] + x[1]*x[1]", degree=2)
    scalar_V = FunctionSpace(mesh, "P", 2)
    scalar_function = interpolate(scalar_expr, scalar_V)

    vector_expr = Expression(("x[0] + x[1]*x[1]", "x[0] - x[1]*x[1]"), degree=2)
    vector_V = VectorFunctionSpace(mesh, "P", 2)
    vector_function = interpolate(vector_expr, vector_V)

    precice_dt = 10
    fenics_dt = 10

    def setUp(self):
        pass

    def test_write_scalar_data(self):
        from precice_future import Interface
        import fenicsadapter
        Interface.configure = MagicMock()
        Interface.write_block_scalar_data = MagicMock()
        Interface.read_block_vector_data = MagicMock(return_value=np.random.rand(self.n_vertices, self.dimensions))
        Interface.get_dimensions = MagicMock(return_value=self.dimensions)
        Interface.set_mesh_vertices = MagicMock(return_value=self.dummy_vertex_ids)
        Interface.initialize = MagicMock(return_value=self.precice_dt)
        Interface.initialize_data = MagicMock()
        Interface.fulfilled_action = MagicMock()
        Interface.is_timestep_complete = MagicMock()
        Interface.advance = MagicMock(return_value=self.precice_dt)
        Interface.get_mesh_id = MagicMock()
        Interface.get_data_id = MagicMock(return_value=15)
        Interface.is_read_data_available = MagicMock(return_value=True)

        write_u = self.scalar_function
        read_u = self.vector_function
        u_init = self.scalar_function

        precice = fenicsadapter.Adapter(self.dummy_config, self.dummy_config)
        precice._interface.writing_initial_data_is_required = MagicMock(return_value=True)
        precice._interface.reading_checkpoint_is_required = MagicMock(return_value=False)
        precice._interface.writing_checkpoint_is_required = MagicMock(return_value=True)
        precice._coupling_bc_expression = MagicMock()
        precice.initialize(RightBoundary(), self.mesh, read_u, write_u, u_init)
        precice.advance(write_u, u_init, u_init, 0, self.fenics_dt, 0)

        expected_data_id = 15
        expected_values = np.array([self.scalar_expr(x_right, y) for y in np.linspace(y_bottom, y_top, 11)])
        expected_ids = self.dummy_vertex_ids

        self.assertTrue(expected_data_id == Interface.write_block_scalar_data.call_args[0][0])
        np.testing.assert_allclose(expected_ids, Interface.write_block_scalar_data.call_args[0][1])
        np.testing.assert_allclose(expected_values, Interface.write_block_scalar_data.call_args[0][2])

    def test_write_vector_data(self):
        from precice_future import Interface
        import fenicsadapter
        Interface.configure = MagicMock()
        Interface.write_block_vector_data = MagicMock()
        Interface.read_block_scalar_data = MagicMock(return_value=np.random.rand(self.n_vertices))
        Interface.get_dimensions = MagicMock(return_value=self.dimensions)
        Interface.set_mesh_vertices = MagicMock(return_value=self.dummy_vertex_ids)
        Interface.initialize = MagicMock(return_value=self.precice_dt)
        Interface.initialize_data = MagicMock()
        Interface.fulfilled_action = MagicMock()
        Interface.is_timestep_complete = MagicMock()
        Interface.advance = MagicMock(return_value=self.precice_dt)
        Interface.get_mesh_id = MagicMock()
        Interface.get_data_id = MagicMock(return_value=15)
        Interface.is_read_data_available = MagicMock(return_value=True)

        write_u = self.vector_function
        read_u = self.scalar_function
        u_init = self.vector_function

        precice = fenicsadapter.Adapter(self.dummy_config, self.dummy_config)
        precice._interface.writing_initial_data_is_required = MagicMock(return_value=True)
        precice._interface.reading_checkpoint_is_required = MagicMock(return_value=False)
        precice._interface.writing_checkpoint_is_required = MagicMock(return_value=True)
        precice._coupling_bc_expression = MagicMock()
        precice.initialize(RightBoundary(), self.mesh, read_u, write_u, u_init)
        precice.advance(write_u, u_init, u_init, 0, self.fenics_dt, 0)

        expected_data_id = 15
        expected_values_x = np.array([write_u(x_right, y)[0] for y in np.linspace(y_bottom, y_top, 11)])
        expected_values_y = np.array([write_u(x_right, y)[1] for y in np.linspace(y_bottom, y_top, 11)])
        expected_values = np.column_stack([expected_values_x, expected_values_y])
        expected_ids = self.dummy_vertex_ids

        self.assertTrue(expected_data_id == Interface.write_block_vector_data.call_args[0][0])
        np.testing.assert_allclose(expected_ids, Interface.write_block_vector_data.call_args[0][1])
        np.testing.assert_allclose(expected_values, Interface.write_block_vector_data.call_args[0][2])

    def test_read_scalar_data(self):
        from precice_future import Interface
        import fenicsadapter

        return_dummy_data = np.arange(self.n_vertices)

        Interface.configure = MagicMock()
        Interface.write_block_vector_data = MagicMock()
        Interface.read_block_scalar_data = MagicMock(return_value=return_dummy_data)
        Interface.get_dimensions = MagicMock(return_value=self.dimensions)
        Interface.set_mesh_vertices = MagicMock(return_value=self.dummy_vertex_ids)
        Interface.initialize = MagicMock(return_value=self.precice_dt)
        Interface.initialize_data = MagicMock()
        Interface.fulfilled_action = MagicMock()
        Interface.is_timestep_complete = MagicMock()
        Interface.advance = MagicMock(return_value=self.precice_dt)
        Interface.get_mesh_id = MagicMock()
        Interface.get_data_id = MagicMock(return_value=15)
        Interface.is_read_data_available = MagicMock(return_value=True)

        write_u = self.vector_function
        read_u = self.scalar_function
        u_init = self.vector_function

        precice = fenicsadapter.Adapter(self.dummy_config, self.dummy_config)
        precice._interface.writing_initial_data_is_required = MagicMock(return_value=True)
        precice._interface.reading_checkpoint_is_required = MagicMock(return_value=False)
        precice._interface.writing_checkpoint_is_required = MagicMock(return_value=True)
        precice._coupling_bc_expression = MagicMock()
        precice.initialize(RightBoundary(), self.mesh, read_u, write_u, u_init)
        precice.advance(write_u, u_init, u_init, 0, self.fenics_dt, 0)

        expected_data_id = 15
        expected_ids = self.dummy_vertex_ids

        self.assertTrue(expected_data_id == Interface.read_block_scalar_data.call_args[0][0])
        np.testing.assert_allclose(expected_ids, Interface.read_block_scalar_data.call_args[0][1])

    def test_read_vector_data(self):
        from precice_future import Interface
        import fenicsadapter

        return_dummy_data = np.random.rand(self.n_vertices, self.dimensions)

        Interface.configure = MagicMock()
        Interface.write_block_scalar_data = MagicMock()
        Interface.read_block_scalar_data = MagicMock()
        Interface.read_block_vector_data = MagicMock(return_value=return_dummy_data)
        Interface.get_dimensions = MagicMock(return_value=self.dimensions)
        Interface.set_mesh_vertices = MagicMock(return_value=self.dummy_vertex_ids)
        Interface.initialize = MagicMock(return_value=self.precice_dt)
        Interface.initialize_data = MagicMock()
        Interface.fulfilled_action = MagicMock()
        Interface.is_timestep_complete = MagicMock()
        Interface.advance = MagicMock(return_value=self.precice_dt)
        Interface.get_mesh_id = MagicMock()
        Interface.get_data_id = MagicMock(return_value=15)
        Interface.is_read_data_available = MagicMock(return_value=True)

        write_u = self.scalar_function
        read_u = self.vector_function
        u_init = self.scalar_function

        precice = fenicsadapter.Adapter(self.dummy_config, self.dummy_config)
        precice._interface.writing_initial_data_is_required = MagicMock(return_value=True)
        precice._interface.reading_checkpoint_is_required = MagicMock(return_value=False)
        precice._interface.writing_checkpoint_is_required = MagicMock(return_value=True)
        precice._coupling_bc_expression = MagicMock()
        precice.initialize(RightBoundary(), self.mesh, read_u, write_u, u_init)
        precice.advance(write_u, u_init, u_init, 0, self.fenics_dt, 0)

        Interface.read_block_scalar_data.assert_not_called()

        expected_data_id = 15
        expected_ids = self.dummy_vertex_ids

        self.assertTrue(expected_data_id == Interface.read_block_vector_data.call_args[0][0])
        np.testing.assert_allclose(expected_ids, Interface.read_block_vector_data.call_args[0][1])
