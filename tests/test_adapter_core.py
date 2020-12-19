from unittest.mock import MagicMock, patch
from unittest import TestCase
import tests.MockedPrecice
from fenics import FunctionSpace, UnitSquareMesh, SubDomain, near, vertices


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
