from unittest.mock import MagicMock, patch
from unittest import TestCase
import tests.MockedPrecice
from fenics import Expression, UnitSquareMesh, FunctionSpace, VectorFunctionSpace, interpolate, SubDomain, near
import numpy as np


@patch.dict('sys.modules', **{'precice': tests.MockedPrecice})
class TestAdapterCore(TestCase):
    def test_get_coupling_boundary_edges(self):
        """
        Test coupling edge detection
        """
        from fenicsprecice.adapter_core import get_coupling_boundary_edges
        from fenics import near, UnitSquareMesh

        class RightBoundary(SubDomain):
            def inside(self, x, on_boundary):
                tol = 1E-14
                if on_boundary and near(x[0], 1, tol):
                    return True
                else:
                    return False

        mesh = UnitSquareMesh(10, 10)  # create dummy mesh
        right_edge = RightBoundary()  # right edge of dummy mesh
        id_mapping = MagicMock()  # a fake id_mapping returning dummy values

        edge_vertex_ids1, edge_vertex_ids2 = get_coupling_boundary_edges(mesh, right_edge, id_mapping)

        self.assertEqual(len(edge_vertex_ids1), 10)
        self.assertEqual(len(edge_vertex_ids2), 10)
