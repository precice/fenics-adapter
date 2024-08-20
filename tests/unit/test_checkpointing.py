from unittest.mock import MagicMock
from unittest import TestCase
from fenics import FunctionSpace, UnitSquareMesh, Expression, interpolate
from fenicsprecice.solverstate import SolverState


class TestCheckpointing(TestCase):
    def test_solverstate_basic(self):
        """
        Check if correct values are read from the checkpoint, while the state of the object that is copied is not changed
        """
        n = 1
        size = 5
        mesh = UnitSquareMesh(size,size)
        V = FunctionSpace(mesh, 'P', 2)
        dummy_value = 1
        E = Expression("t", t=dummy_value, degree=2)
        u = interpolate(E, V)

        # "write checkpoint"
        sstate = SolverState(u, dummy_value, n)
        # "read checkpoint"
        u_cp, t_cp, n_cp = sstate.get_state()

        #check values
        self.assertEqual(t_cp, dummy_value)
        self.assertEqual(n, n_cp)
        #function should be the same everywhere (-> check vector values of the function)
        vec_u = u.vector()
        vec_u_cp = u_cp.vector()
        for i in range(size*size):
            self.assertAlmostEqual(vec_u[i], vec_u_cp[i])

    def test_solverstate_modification_vector(self):
        """
        Check if correct values are read from the checkpoint, if the dof vector of the dolfin functions are changed directly 

        Motivation for this test: Related to https://github.com/precice/fenics-adapter/pull/172 and https://github.com/precice/tutorials/pull/554
        """
        n = 1
        size = 5
        mesh = UnitSquareMesh(size,size)
        V = FunctionSpace(mesh, 'P', 2)
        ref_value = 1
        E = Expression("t", t=ref_value, degree=2)
        u = interpolate(E, V)

        # "write checkpoint"
        sstate = SolverState(u, ref_value, n)

        # modify state of u
        dummy_value = ref_value + 2
        u.vector()[:] = dummy_value

        # "read checkpoint"
        u_cp, _, _ = sstate.get_state()

        #check values
        #function should be the same everywhere
        #(so the vector values should all be ref_value)
        vec_u_cp = u_cp.vector()
        for i in range(size*size):
            self.assertAlmostEqual(ref_value, vec_u_cp[i])
    

    def test_solverstate_modification_assign(self):
        """
        Check if correct values are read from the checkpoint, if the dof of the dolfin functions are changed with the assign function
        and not directly via the dof vector
        """
        n = 1
        size = 5
        mesh = UnitSquareMesh(size,size)
        V = FunctionSpace(mesh, 'P', 2)
        ref_value = 1
        E = Expression("t", t=ref_value, degree=2)
        u = interpolate(E, V)

        # "write checkpoint"
        sstate = SolverState(u, ref_value, n)

        # modify state of u
        # "compute" new solution
        dummy_value = ref_value + 2
        E.t = dummy_value
        u2 = interpolate(E,V)
        u.assign(u2)

        # "read checkpoint"
        u_cp, _, _ = sstate.get_state()

        #check values
        #function should be the same everywhere
        #(so the vector values should all be ref_value)
        vec_u_cp = u_cp.vector()
        for i in range(size*size):
            self.assertAlmostEqual(ref_value, vec_u_cp[i])
