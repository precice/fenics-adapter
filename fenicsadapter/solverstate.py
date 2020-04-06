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
        return self.u.copy(), self.t, self.n

    def print_state(self):
        u, t, n = self.get_state()
        return print("u={u}, t={t}, n={n}".format(u=u, t=t, n=n))
