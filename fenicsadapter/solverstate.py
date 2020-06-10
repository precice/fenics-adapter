class SolverState:
    def __init__(self, u, t, n):
        """
        Solver state consists of a value u, associated time t and the timestep n

        Parameters
        ----------
        u : Object of class dolfin.functions.function.Function
            FEniCS function related to the field during each coupling iteration.
        t : double
            Time stamp.
        n : int
            Iteration number.
        """
        self.u = u
        self.t = t
        self.n = n

    def get_state(self):
        """
        Returns the state variables value u, associated time t and timestep n

        Returns
        -------
        u : Object of class dolfin.functions.function.Function
            A copy of FEniCS function related to the field during each coupling iteration.
        t : double
            Time stamp.
        n : int
            Iteration number.
        """
        return self.u.copy(), self.t, self.n

    def print_state(self):
        u, t, n = self.get_state()
        return print("u={u}, t={t}, n={n}".format(u=u, t=t, n=n))
