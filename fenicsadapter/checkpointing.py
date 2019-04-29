class Checkpoint:

    def __init__(self):
        """
        A checkpoint for the solver state
        """
        self._u = None  # checkpoint for function value inside domain
        self._t = None  # time of the checkpoint
        self._n = None  # timestep of the checkpoint

    def read(self, u):
        """
        read solver state from checkpoint to u
        :param u: function value, which is set to checkpoint value (call-by-reference)
        :return: checkpoint time and timestep
        """
        u.assign(self._u)  # set u to value of checkpoint
        return self._t, self._n

    def write(self, u, t, n):
        """
        write checkpoint from solver state.
        :param u: function value
        :param t: time
        :param n: timestep
        """
        if not self.is_empty():
            self._u.assign(u)
        else:
            self._u = u.copy()
        self._t = t
        self._n = n

    def is_empty(self):
        """
        Returns whether checkpoint is empty. An empty checkpoint has the function value self._u from self.__init__
        :return:
        """
        checkpoint_is_empty = not self._u
        if checkpoint_is_empty:  # if checkpoint_is_empty, assert that self._t and self._n are also None
            assert (not self._t)
            assert (not self._n)

        return checkpoint_is_empty
