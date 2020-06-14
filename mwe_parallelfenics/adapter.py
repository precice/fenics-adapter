"""
Dummy adapter for MWE
"""


class Adapter:
    def __init__(self):
        self._expr = None

    def set_func(self, expr):
        self._expr = expr

    def get_func(self):
        return self._expr
