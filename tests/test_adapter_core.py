from unittest.mock import MagicMock, patch
from unittest import TestCase
import tests.MockedPrecice
import numpy as np
from fenics import FunctionSpace, VectorFunctionSpace,UnitSquareMesh, SubDomain, near, vertices, Expression, interpolate


@patch.dict('sys.modules', **{'precice': tests.MockedPrecice})
class TestAdapterCore(TestCase):
    def test_get_coupling_boundary_edges(self):
        """
        Test coupling edge detection
        """
        from fenicsprecice.adapter_core import get_coupling_boundary_edges

        class RightBoundary(SubDomain):
            def inside(self, x, on_boundary):
                tol = 1E-14
                if on_boundary and near(x[0], 1, tol):
                    return True
                else:
                    return False

        mesh = UnitSquareMesh(10, 10)  # create dummy mesh
        V = FunctionSpace(mesh, 'P', 2)  # Create function space using mesh
        right_edge = RightBoundary()  # right edge of dummy mesh
        id_mapping = MagicMock()  # a fake id_mapping returning dummy values

        global_ids = []
        for v in vertices(mesh):
            if right_edge.inside(v.point(), True):
                global_ids.append(v.global_index())

        edge_vertex_ids1, edge_vertex_ids2 = get_coupling_boundary_edges(V, right_edge, global_ids, id_mapping)

        self.assertEqual(len(edge_vertex_ids1), 10)
        self.assertEqual(len(edge_vertex_ids2), 10)

    def test_convert_fenics_to_precice(self):
        """
        Test conversion from function to write_data
        """
        from fenicsadapter.adapter_core import convert_fenics_to_precice        
        from sympy import lambdify, symbols, printing

        mesh = UnitSquareMesh(10, 10)  # create dummy mesh

        # scalar valued
        V = FunctionSpace(mesh, 'P', 2)  # Create function space using mesh
        x, y = symbols('x[0], x[1]')
        fun_sym = y + x*x
        fun_lambda = lambdify([x,y], fun_sym)
        fun_string = printing.ccode(fun_sym)
        expression = Expression(fun_string, degree=2)
        fenics_function = interpolate(expression, V)
        
        local_ids = []
        manual_sampling = []
        for v in vertices(mesh):
            local_ids.append(v.index())
            manual_sampling.append(fun_lambda(v.x(0), v.x(1)))

        data = convert_fenics_to_precice(fenics_function, local_ids)

        np.testing.assert_allclose(data, manual_sampling)

        # vector valued
        W = VectorFunctionSpace(mesh, 'P', 2)  # Create function space using mesh
        fun_sym_x = y + x*x
        fun_sym_y = y*y + x*x*x*2
        fun_lambda = lambdify([x,y], [fun_sym_x, fun_sym_y])
        fun_string = (printing.ccode(fun_sym_x), printing.ccode(fun_sym_y))
        expression = Expression(fun_string, degree=2)
        fenics_function = interpolate(expression, W)

        local_ids = []
        manual_sampling = []
        for v in vertices(mesh):
            local_ids.append(v.index())
            manual_sampling.append(fun_lambda(v.x(0), v.x(1)))

        data = convert_fenics_to_precice(fenics_function, local_ids)

        np.testing.assert_allclose(data, manual_sampling)