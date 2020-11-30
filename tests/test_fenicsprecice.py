# comments on test layout: https://docs.pytest.org/en/latest/goodpractices.html

from unittest.mock import MagicMock, patch
from unittest import TestCase
import tests.MockedPrecice
import numpy as np
from fenics import Expression, UnitSquareMesh, FunctionSpace, VectorFunctionSpace, interpolate, dx, ds, \
    SubDomain, near, PointSource, Point, AutoSubDomain, TestFunction, \
    grad, assemble, Function, solve, dot

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
class TestAdapter(TestCase):
    """
    Test suite for basic API functions
    """
    def test_version(self):
        """
        Test that adapter provides a version
        """
        import fenicsprecice
        fenicsprecice.__version__


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
        import fenicsprecice
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

        precice = fenicsprecice.Adapter(self.dummy_config)

        precice.store_checkpoint(self.u_n_mocked, self.t, self.n)

        # Replicating control flow where implicit iteration has not converged and solver state needs to be restored
        # to a checkpoint
        precice.advance(self.dt)
        Interface.is_time_window_complete = MagicMock(return_value=False)

        # Check if the checkpoint is stored correctly in the adapter
        self.assertEqual(precice.retrieve_checkpoint() == self.u_n_mocked, self.t, self.n)


@patch.dict('sys.modules', **{'precice': tests.MockedPrecice})
class TestExpressionHandling(TestCase):
    """
    Test Expression creation and updating mechanism based on data provided by user.
    """
    dummy_config = "tests/precice-adapter-config.json"

    mesh = UnitSquareMesh(10, 10)
    dimension = 2

    scalar_expr = Expression("x[0] + x[1]", degree=1)
    scalar_V = FunctionSpace(mesh, "P", 1)
    scalar_function = interpolate(scalar_expr, scalar_V)

    vector_expr = Expression(("x[0] + x[1]*x[1]", "x[0] - x[1]*x[1]"), degree=2)
    vector_V = VectorFunctionSpace(mesh, "P", 2)
    vector_function = interpolate(vector_expr, vector_V)

    n_vertices = 11
    fake_id = 15
    vertices_x = [1 for _ in range(n_vertices)]
    vertices_y = np.linspace(0, 1, n_vertices)
    vertex_ids = np.arange(n_vertices)

    n_samples = 1000
    samplepts_x = [1 for _ in range(n_samples)]
    samplepts_y = np.linspace(0, 1, n_samples)

    class Right(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 1.0)

    def test_update_expression_scalar(self):
        """
        Check if a sampling of points on a dolfin Function interpolated via FEniCS is matching with the sampling of the
        same points on a FEniCS Expression created by the Adapter
        """
        from precice import Interface
        import fenicsprecice

        Interface.get_dimensions = MagicMock(return_value=2)
        Interface.set_mesh_vertices = MagicMock(return_value=self.vertex_ids)
        Interface.get_mesh_id = MagicMock()
        Interface.get_data_id = MagicMock()
        Interface.set_mesh_edge = MagicMock()
        Interface.initialize = MagicMock()
        Interface.initialize_data = MagicMock()
        Interface.is_action_required = MagicMock()
        Interface.mark_action_fulfilled = MagicMock()
        Interface.write_block_scalar_data = MagicMock()

        right_boundary = self.Right()

        precice = fenicsprecice.Adapter(self.dummy_config)
        precice._interface = Interface(None, None, None, None)
        precice.initialize(right_boundary, self.scalar_V, self.scalar_function)
        values = np.array([self.scalar_function(x, y) for x, y in zip(self.vertices_x, self.vertices_y)])
        data = {(x, y): v for x, y, v in zip(self.vertices_x, self.vertices_y, values)}

        scalar_coupling_expr = precice.create_coupling_expression()
        precice.update_coupling_expression(scalar_coupling_expr, data)

        expr_samples = np.array([scalar_coupling_expr(x, y) for x, y in zip(self.samplepts_x, self.samplepts_y)])
        func_samples = np.array([self.scalar_function(x, y) for x, y in zip(self.samplepts_x, self.samplepts_y)])

        assert (np.allclose(expr_samples, func_samples, 1E-10))

    def test_update_expression_vector(self):
        """
        Check if a sampling of points on a dolfin Function interpolated via FEniCS is matching with the sampling of the
        same points on a FEniCS Expression created by the Adapter
        """
        from precice import Interface
        import fenicsprecice

        Interface.get_dimensions = MagicMock(return_value=2)
        Interface.set_mesh_vertices = MagicMock(return_value=self.vertex_ids)
        Interface.get_mesh_id = MagicMock()
        Interface.get_data_id = MagicMock()
        Interface.set_mesh_edge = MagicMock()
        Interface.initialize = MagicMock()
        Interface.initialize_data = MagicMock()
        Interface.is_action_required = MagicMock()
        Interface.mark_action_fulfilled = MagicMock()
        Interface.write_block_vector_data = MagicMock()

        right_boundary = self.Right()

        precice = fenicsprecice.Adapter(self.dummy_config)
        precice._interface = Interface(None, None, None, None)
        precice.initialize(right_boundary, self.vector_V, self.vector_function)
        values = np.array([self.vector_function(x, y) for x, y in zip(self.vertices_x, self.vertices_y)])
        data = {(x, y): v for x, y, v in zip(self.vertices_x, self.vertices_y, values)}

        vector_coupling_expr = precice.create_coupling_expression()
        precice.update_coupling_expression(vector_coupling_expr, data)

        expr_samples = np.array([vector_coupling_expr(x, y) for x, y in zip(self.samplepts_x, self.samplepts_y)])
        func_samples = np.array([self.vector_function(x, y) for x, y in zip(self.samplepts_x, self.samplepts_y)])

        assert (np.allclose(expr_samples, func_samples, 1E-10))


# TODO: Write a valid test for adapter function get_point_sources. Direct comparison of PointSource object is not
#       possible because objects initialized by same force values are still different. No preCICE API call is done
#       within this function so comparing input parameters to an API call is also not possible.
#       Possible solution: Solve a 2D beam problem manually and via adapter function and check displacement of the beam.
#                           Then the question is whether it is a mock test at all.
# def clamped_boundary(x, on_boundary):
#     return on_boundary and abs(x[1]) < 1E-14
#
# @patch.dict('sys.modules', **{'precice': tests.MockedPrecice})
# class TestPointSource(TestCase):
#     """
#     Test Point Source return mechanism for force vector values given by user.
#     """
#     dummy_config = "tests/precice-adapter-config.json"
#
#     mesh = UnitSquareMesh(10, 10)
#     dimension = 2
#     V = VectorFunctionSpace(mesh, "P", 2)
#
#     n_vertices = 11
#     fake_id = 15
#     vertices_x = [1 for _ in range(n_vertices)]
#     vertices_y = np.linspace(0, 1, n_vertices)
#     vertex_ids = np.arange(n_vertices)
#     vertices = []
#     for i in range(n_vertices):
#         vertices.append([vertices_x[i], vertices_y[i]])
#     vertices = np.array(vertices)
#
#     def test_get_point_sources(self):
#         """
#         Checks
#         Returns
#         -------
#         """
#         from precice import Interface
#         import fenicsprecice
#         from fenicsprecice.adapter_core import FunctionType
#
#         Interface.get_dimensions = MagicMock(return_value=2)
#         Interface.set_mesh_vertices = MagicMock(return_value=self.vertex_ids)
#         Interface.get_mesh_id = MagicMock()
#         Interface.set_mesh_edge = MagicMock()
#         Interface.initialize = MagicMock()
#
#         # Define 2D dummy forces
#         dummy_forces_array = []
#         for i in range(self.n_vertices):
#             dummy_forces_array.append([i, i])
#         dummy_forces_array = np.array(dummy_forces_array)
#
#         # Define Point Sources manually
#         f_x_manual = dict()
#         f_y_manual = dict()
#         for i in range(self.n_vertices):
#             px, py = self.vertices_x[i], self.vertices_y[i]
#             key = (px, py)
#             f_x_manual[key] = PointSource(self.V.sub(0), Point(px, py), dummy_forces_array[i, 0])
#             f_y_manual[key] = PointSource(self.V.sub(1), Point(px, py), dummy_forces_array[i, 1])
#
#         precice = fenicsprecice.Adapter(self.dummy_config)
#         precice._function_space = self.V
#         precice._Dirichlet_Boundary = AutoSubDomain(clamped_boundary)
#         precice._read_function_type = FunctionType.VECTOR
#         precice._fenics_dimensions = self.dimension
#         precice._coupling_mesh_vertices = self.vertices
#
#         # Define same dummy forces as a dictionary
#         dummy_forces = dict()
#         counter = 0
#         for v in self.vertices:
#             dummy_forces[tuple(v)] = [counter, counter]
#             counter += 1
#
#         # Get forces from Adapter function
#         f_x_adapter, f_y_adapter = precice.get_point_sources(dummy_forces)
#
#         v = TestFunction(self.V)
#
#         a = assemble(dot(grad(v), grad(v))*dx)
#         b_manual = assemble(v*dx)
#         b_adapter = assemble(v*dx)
#
#         u_manual = Function(self.V)
#         u_adapter = Function(self.V)
#
#         for ps in f_x_manual.values():
#             ps.apply(b_manual)
#         for ps in f_y_manual.values():
#             ps.apply(b_manual)
#
#         for ps in x_forces:
#             ps.apply(b_adapter)
#         for ps in y_forces:
#             ps.apply(b_adapter)
#
#         solve(a, u_manual.vector(), b_manual)
#         solve(a, u_adapter.vector(), b_adapter)
#
#         assert(u_manual == u_adapter)

