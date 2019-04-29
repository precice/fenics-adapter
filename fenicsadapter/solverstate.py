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
        updates the state using FEniCS assing function
        :param other_state:
        :return:
        """
        self.u.assign(other_state.u)
        self.t = other_state.t
        self.n = other_state.n
