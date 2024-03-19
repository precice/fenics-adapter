class SolverState:
    def __init__(self, payload, t, n):
        """
        Solver state consists of a payload (either a single fenics.Function or a list of several fenics.Functions), associated time t and the timestep n

        Parameters
        ----------
        payload : A fenics.Function or a list of fenics.Functions
            Describes the state of the solver.
        t : double
            Time stamp.
        n : int
            Iteration number.
        """
        try:
            self.payload = payload.copy()
        except AttributeError:  # if .copy() does not exist, it probably is a list
            self.payload = [item.copy() for item in payload]

        self.t = t
        self.n = n

    def get_state(self):
        """
        Returns the state variables payload, associated time t and timestep n

        Returns
        -------
        payload : A fenics.Function or a list of fenics.Functions
            Describes the state of the solver.
        t : double
            Time stamp.
        n : int
            Iteration number.
        """
        try:
            return self.payload.copy(), self.t, self.n
        except AttributeError:  # if .copy() does not exist, it probably is a list
            return [item.copy() for item in self.payload], self.t, self.n

    def print_state(self):
        payload, t, n = self.get_state()
        return print(f"payload={payload}, t={t}, n={n}")
