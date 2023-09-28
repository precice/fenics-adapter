# comments on test layout: https://docs.pytest.org/en/latest/goodpractices.html

from unittest.mock import MagicMock, patch
from unittest import TestCase
from tests import MockedPrecice
import numpy as np
from fenics import Expression, UnitSquareMesh, FunctionSpace, VectorFunctionSpace, interpolate, dx, \
    SubDomain, near, PointSource, Point, AutoSubDomain, TestFunction, grad, dot, TrialFunction, \
    TestFunction, inner, Constant, assemble_system


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


@patch.dict('sys.modules', {'precice': MockedPrecice})
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


@patch.dict('sys.modules', {'precice': MockedPrecice})
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
        from precice import Participant

        Participant.initialize = MagicMock(return_value=self.dt)
        Participant.get_mesh_dimensions = MagicMock()
        Participant.is_time_window_complete = MagicMock(return_value=True)
        Participant.advance = MagicMock()

        precice = fenicsprecice.Adapter(self.dummy_config)

        precice.store_checkpoint(self.u_n_mocked, self.t, self.n)

        # Replicating control flow where implicit iteration has not converged and solver state needs to be restored
        # to a checkpoint
        precice.advance(self.dt)
        Participant.is_time_window_complete = MagicMock(return_value=False)

        # Check if the checkpoint is stored correctly in the adapter
        self.assertEqual(precice.retrieve_checkpoint() == self.u_n_mocked, self.t, self.n)


@patch.dict('sys.modules', {'precice': MockedPrecice})
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
    vertices_x = [1 for _ in range(n_vertices)]
    vertices_y = np.linspace(0, 1, n_vertices)
    vertex_ids = np.arange(n_vertices)

    n_samples = 1000
    samplepts_x = [1 for _ in range(n_samples)]
    samplepts_y = np.linspace(0, 1, n_samples)

    class Right(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 1.0)

    def test_create_expression_scalar(self):
        """
        Check if a sampling of points on a dolfin Function interpolated via FEniCS is matching with the sampling of the
        same points on a FEniCS Expression created by the Adapter
        """
        from precice import Participant
        import fenicsprecice

        Participant.get_mesh_dimensions = MagicMock(return_value=2)
        Participant.set_mesh_vertices = MagicMock(return_value=self.vertex_ids)
        Participant.requires_mesh_connectivity_for = MagicMock(return_value=False)
        Participant.requires_initial_data = MagicMock(return_value=False)
        Participant.initialize = MagicMock()
        Participant.write_data = MagicMock()

        right_boundary = self.Right()

        precice = fenicsprecice.Adapter(self.dummy_config)
        precice._participant = Participant(None, None, None, None)
        precice.initialize(right_boundary, self.scalar_V, self.scalar_function)
        precice.create_coupling_expression()
        # currently only a smoke tests. Is there a good way to test this?

    def test_update_expression_scalar(self):
        """
        Check if a sampling of points on a dolfin Function interpolated via FEniCS is matching with the sampling of the
        same points on a FEniCS Expression created by the Adapter
        """
        from precice import Participant
        import fenicsprecice

        Participant.get_mesh_dimensions = MagicMock(return_value=2)
        Participant.set_mesh_vertices = MagicMock(return_value=self.vertex_ids)
        Participant.requires_mesh_connectivity_for = MagicMock(return_value=False)
        Participant.requires_initial_data = MagicMock(return_value=False)
        Participant.initialize = MagicMock()
        Participant.write_data = MagicMock()

        right_boundary = self.Right()

        precice = fenicsprecice.Adapter(self.dummy_config)
        precice._participant = Participant(None, None, None, None)
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
        from precice import Participant
        import fenicsprecice

        Participant.get_mesh_dimensions = MagicMock(return_value=2)
        Participant.set_mesh_vertices = MagicMock(return_value=self.vertex_ids)
        Participant.requires_mesh_connectivity_for = MagicMock(return_value=False)
        Participant.requires_initial_data = MagicMock(return_value=False)
        Participant.initialize = MagicMock()
        Participant.write_data = MagicMock()

        right_boundary = self.Right()

        precice = fenicsprecice.Adapter(self.dummy_config)
        precice._participant = Participant(None, None, None, None)
        precice.initialize(right_boundary, self.vector_V, self.vector_function)
        values = np.array([self.vector_function(x, y) for x, y in zip(self.vertices_x, self.vertices_y)])
        data = {(x, y): v for x, y, v in zip(self.vertices_x, self.vertices_y, values)}

        vector_coupling_expr = precice.create_coupling_expression()
        precice.update_coupling_expression(vector_coupling_expr, data)

        expr_samples = np.array([vector_coupling_expr(x, y) for x, y in zip(self.samplepts_x, self.samplepts_y)])
        func_samples = np.array([self.vector_function(x, y) for x, y in zip(self.samplepts_x, self.samplepts_y)])

        assert (np.allclose(expr_samples, func_samples, 1E-10))


@patch.dict('sys.modules', **{'precice': MockedPrecice})
class TestPointSource(TestCase):
    """
    Test Point Source return mechanism for force vector values given by user.
    """
    dummy_config = "tests/precice-adapter-config.json"

    def test_get_point_sources_2d(self):
        """
        Checks
        Returns
        -------
        """
        import fenicsprecice
        from fenicsprecice.adapter_core import FunctionType, filter_point_sources

        n_vertices = 11
        dimensions = 2
        mesh = UnitSquareMesh(10, 10)
        V = VectorFunctionSpace(mesh, "P", 2)

        u = TrialFunction(V)
        v = TestFunction(V)
        f = Constant((0, 0))

        a = inner(grad(u), grad(v)) * dx
        L = dot(f, v) * dx
        _, b = assemble_system(a, L)

        vertices_x = [1 for _ in range(n_vertices)]
        vertices_y = np.linspace(0, 1, n_vertices)
        vertices = []
        for i in range(n_vertices):
            vertices.append([vertices_x[i], vertices_y[i]])
        vertices = np.array(vertices)

        # Use this to fix the x=0 boundary of the domain to 0
        def dirichlet_boundary(x, on_boundary): return on_boundary and abs(x[0]) < 1E-14
        fixed_boundary = AutoSubDomain(dirichlet_boundary)

        precice = fenicsprecice.Adapter(self.dummy_config)
        precice._read_function_space = V
        precice._Dirichlet_Boundary = AutoSubDomain(dirichlet_boundary)
        precice._read_function_type = FunctionType.VECTOR

        # Define 2D dummy forces
        dummy_nodal_data = []
        for i in range(n_vertices):
            dummy_nodal_data.append([i, i])

        dummy_nodal_data = np.array(dummy_nodal_data)

        data = {tuple(key): value for key, value in zip(vertices, dummy_nodal_data)}

        dummy_forces = []
        for d in range(dimensions):
            dummy_forces.append(dict())

        for i in range(n_vertices):
            key = []
            for d in range(dimensions):
                key.append(vertices[i, d])
            key = tuple(key)

            for d in range(dimensions):
                dummy_forces[d][key] = PointSource(V.sub(d), Point(key), dummy_nodal_data[i, d])

        # Avoid application of PointSource and Dirichlet boundary condition at the same point by filtering
        for d in range(dimensions):
            dummy_forces[d] = filter_point_sources(dummy_forces[d], fixed_boundary, warn_duplicate=False)

        # Get forces from Adapter function
        forces_x, forces_y = precice.get_point_sources(data)

        b_dummy = b.copy()
        b_forces = b.copy()

        for ps in dummy_forces[0].values():
            ps.apply(b_dummy)
        for ps in dummy_forces[1].values():
            ps.apply(b_dummy)

        for ps in forces_x:
            ps.apply(b_forces)
        for ps in forces_y:
            ps.apply(b_forces)

        assert (np.allclose(b_dummy.get_local(), b_forces.get_local()))
