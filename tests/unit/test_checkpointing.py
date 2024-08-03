from unittest.mock import MagicMock
from unittest import TestCase
from fenics import FunctionSpace, UnitSquareMesh, Expression, interpolate
from fenicsprecice.solverstate import SolverState


class TestCheckpointing(TestCase):
    def test_solverstate_basic(self):
        """
        Check if correct values are read from the checkpoint, while the state of the object that is copied is not changed
        """
        time = 0.5
        iteration = 1
        size = 5
        mesh = UnitSquareMesh(size,size)
        V = FunctionSpace(mesh, 'P', 2)
        E = Expression("t", t=time, degree=2)
        u = interpolate(E, V)

        # "write checkpoint"
        sstate = SolverState(u, time, iteration)
        # "read checkpoint"
        u_cp, t_cp, n_cp = sstate.get_state()

        #check values
        self.assertEqual(t_cp, time)
        self.assertEqual(iteration, n_cp)
        #function should be the same everywhere (-> check vector values of the function)
        vec_u = u.vector()
        vec_u_cp = u_cp.vector()
        for i in range(size*size):
            self.assertAlmostEqual(vec_u[i], vec_u_cp[i])

    def test_solverstate_modification_vector(self):
        """
        Check if correct values are read from the checkpoint, if the dof vector of the dolfin functions are changed directly 
        """
        time = 0.5
        iteration = 1
        size = 5
        mesh = UnitSquareMesh(size,size)
        V = FunctionSpace(mesh, 'P', 2)
        E = Expression("t", t=time, degree=2)
        u = interpolate(E, V)

        # "write checkpoint"
        sstate = SolverState(u, time, iteration)

        # modify state of u
        u.vector()[:] = time + 2

        # "read checkpoint"
        u_cp, _, _ = sstate.get_state()

        #check values
        #function should be the same everywhere
        #(so the vector values should all be time=0.5)
        vec_u_cp = u_cp.vector()
        for i in range(size*size):
            self.assertAlmostEqual(time, vec_u_cp[i])
    

    def test_solverstate_modification_assign(self):
        """
        Check if correct values are read from the checkpoint, if the dof of the dolfin functions are changed with the assign function
        and not directly via the dof vector
        """
        time = 0.5
        iteration = 1
        size = 5
        mesh = UnitSquareMesh(size,size)
        V = FunctionSpace(mesh, 'P', 2)
        E = Expression("t", t=time, degree=2)
        u = interpolate(E, V)

        # "write checkpoint"
        sstate = SolverState(u, time, iteration)

        # modify state of u
        # "compute" new solution
        E.t += 2
        u2 = interpolate(E,V)
        u.assign(u2)

        # "read checkpoint"
        u_cp, _, _ = sstate.get_state()

        #check values
        #function should be the same everywhere
        #(so the vector values should all be time=0.5)
        vec_u_cp = u_cp.vector()
        for i in range(size*size):
            self.assertAlmostEqual(time, vec_u_cp[i])
