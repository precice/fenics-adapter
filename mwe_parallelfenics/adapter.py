"""
Dummy adapter for MWE
"""
import dolfin
import numpy as np


class Adapter:
    def __init__(self):
        self._function = None

    def eval_func(self, function, points):
        if type(function) is dolfin.Function:
            x_all, y_all = points[:, 0], points[:, 1]
            for x, y in zip(x_all, y_all):
                print("(x,y) = ({},{})".format(x, y))
                print("function evaluation at ({},{}) = {}".format(x, y, function(x, y)))

