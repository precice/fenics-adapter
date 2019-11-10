def check_type_consistency(other_u, my_u):
    try:
        assert(type(other_u) == type(my_u))
    except AssertionError as e:
        raise Exception("{}: other_u = {}; my_u = {}".format(e, other_u, my_u))


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

    def _is_initialized(self):
        return (self.u is not None) and (self.t is not None) and (self.n is not None)

    def get_state(self):
        """
        returns the state variables value u, associated time t and timestep n
        :return:
        """
        return self.u, self.t, self.n

    def update(self, other_state):
        """
        updates the state using FEniCS assing function. self.u is updated.
        This may also have an effect outside of this object! Compare to SolverState.copy(other_state).
        :param other_state:
        """
        assert(self._is_initialized())
        check_type_consistency(other_state.u, self.u)

        if type(other_state.u) is tuple:
            for elem, other_elem in zip(self.u, other_state.u):
                elem.assign(other_elem)
        else:
            self.u.assign(other_state.u)

        self.t = other_state.t
        self.n = other_state.n

    def get_copy(self):
        if type(self.u) is tuple:
            copied_u = tuple((elem.copy() for elem in self.u))
        else:
            copied_u = self.u.copy()

        copied_t = self.t
        copied_n = self.n

        return SolverState(copied_u, copied_t, copied_n)

    def copy(self, other_state):
        """
        copies a state using FEniCS copy function. self.u is overwritten.
        This does not have an effect outside of this object! Compare to SolverState.update(other_state).
        :param other_state:
        """
        assert(self._is_initialized())
        check_type_consistency(other_state.u, self.u)

        copied_state = other_state.get_copy()
        self.u = copied_state.u
        self.t = copied_state.t
        self.n = copied_state.n

    def print_state(self):
        u, t, n = self.get_state()
        return "u={u}, t={t}, n={n}".format(u=u, t=t, n=n)
