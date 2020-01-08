class SolverState:
    def __init__(self, u, t, n):
        """
        Solver state consists of a value u, associated time t and the timestep n
        :param u: value
        :param t: time
        :param n: timestep
        """
        self.u = u
        self.t = t
        self.n = n

    def get_state(self):
        """
        returns the state variables value u, associated time t and timestep n
        :return:
        """
        return self.u, self.t, self.n

    def update(self, other_state):
        """
        updates the state using FEniCS assign function. self.u is updated.
        This may also have an effect outside of this object! Compare to SolverState.copy(other_state).
        :param other_state:
        """
        self.u.assign(other_state.u)
        self.t = other_state.t
        self.n = other_state.n

    def copy(self, other_state):
        """
        copies a state using FEniCS copy function. self.u is overwritten.
        This does not have an effect outside of this object! Compare to SolverState.update(other_state).
        :param other_state:
        """
        self.u = other_state.u.copy()
        self.t = other_state.t
        self.n = other_state.n

    def print_state(self):
        u, t, n = self.get_state()
        return "u={u}, t={t}, n={n}".format(u=u, t=t, n=n)
