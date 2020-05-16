"""
Provide two types of FEniCS Expressions as coupling boundary Expressions to the user.
"""

import dolfin
from dolfin import UserExpression
from scipy.interpolate import Rbf
from scipy.interpolate import interp1d
import numpy as np
from fenics import MPI

import logging

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class CustomExpression(UserExpression):
    """Creates functional representation (for FEniCS) of nodal data
    provided by preCICE.
    """

    def update_boundary_data(self, vals, coords_x, coords_y=None, coords_z=None):
        """ update the data stored by expression.

        :param vals: data values on nodes
        :param coords_x: x coordinates of nodes
        :param coords_y: y coordinates of nodes
        :param coords_z: z coordinates of nodes
        """

        self._coords_x = coords_x
        self._dimension = 3
        if coords_y is None:
            self._dimension -= 1
            coords_y = np.zeros(self._coords_x.shape)
        self._coords_y = coords_y
        if coords_z is None:
            self._dimension -= 1
            coords_z = np.zeros(self._coords_x.shape)

        self._coords_y = coords_y
        self._coords_z = coords_z
        self._vals = vals

        self._f = self.create_interpolant()

        if self.is_scalar_valued():
            assert (self._vals.shape == self._coords_x.shape)
        elif self.is_vector_valued():
            assert (self._vals.shape[0] == self._coords_x.shape[0])

    def interpolate(self, x):
        # TODO: the correct way to deal with this would be using an abstract class. Since this is technically more complex and the current implementation is a workaround anyway, we do not use the proper solution, but this hack.
        """ Interpolates at x. Uses buffered interpolant self._f.

        :return: returns a list containing the interpolated values. If scalar function is interpolated this list has a
        single element. If a vector function is interpolated the list has self._dimensions elements.
        """
        raise Exception("Please use one of the classes derived from this class, that implements an actual strategy for"
                        "interpolation.")

    def create_interpolant(self, x):
        # TODO: the correct way to deal with this would be using an abstract class. Since this is technically more complex and the current implementation is a workaround anyway, we do not use the proper solution, but this hack.
        """ Creates interpolant from boundary data that has been provided before.

        :return: returns interpolant as list. If scalar function is interpolated this list has a single element. If a
        vector function is interpolated the list has self._dimensions elements.
        """
        raise Exception("Please use one of the classes derived from this class, that implements an actual strategy for"
                        "interpolation.")

    def eval(self, value, x):
        """ Evaluates expression at x using self.interpolate(x) and stores result to value.

        :param x: coordinate where expression has to be evaluated
        :param value: buffer where result has to be returned to
        """
        return_value = self.interpolate(x)
        for i in range(self._vals.ndim):
            value[i] = return_value[i]

    def is_scalar_valued(self):
        """ Determines if function being interpolated is scalar-valued based on dimension of provided vector self._vals.

        :return: whether function is scalar valued
        """
        if self._vals.ndim == 1:
            return True
        elif self._vals.ndim > 1:
            return False
        else:
            raise Exception("Dimension of the function is 0 or negative!")

    def is_vector_valued(self):
        """ Determines if function being interpolated is vector-valued based on dimension of provided vector self._vals.

        :return: whether function is scalar valued
        """
        if self._vals.ndim > 1:
            return True
        elif self._vals.ndim == 1:
            return False
        else:
            raise Exception("Dimension of the function is 0 or negative!")


class GeneralInterpolationExpression(CustomExpression):
    """Uses RBF interpolation for implementation of CustomExpression.interpolate. Allows for arbitrary coupling
    interfaces, but has limited accuracy.
    """

    def create_interpolant(self):
        interpolant = []
        if self._dimension == 1:
            assert (
                self.is_scalar_valued())  # for 1D only R->R mapping is allowed by preCICE, no need to implement Vector case
            interpolant.append(Rbf(self._coords_x, self._vals.flatten()))
        elif self._dimension == 2:
            if self.is_scalar_valued():  # check if scalar or vector-valued
                interpolant.append(Rbf(self._coords_x, self._coords_y, self._vals.flatten()))
            elif self.is_vector_valued():
                interpolant.append(Rbf(self._coords_x, self._coords_y,
                                       self._vals[:, 0].flatten()))  # extract dim_no element of each vector
                interpolant.append(Rbf(self._coords_x, self._coords_y,
                                       self._vals[:, 1].flatten()))  # extract dim_no element of each vector
            else:
                raise Exception("Problem dimension and data dimension not matching.")
        elif self._dimension == 3:
            logger.warning("RBF Interpolation for 3D Simulations has not been properly tested!")
            if self.is_scalar_valued():
                interpolant.append(Rbf(self._coords_x, self._coords_y, self._coords_z, self._vals.flatten()))
            elif self.is_vector_valued():
                interpolant.append(Rbf(self._coords_x, self._coords_y, self._coords_z, self._vals[:, 0].flatten()))
                interpolant.append(Rbf(self._coords_x, self._coords_y, self._coords_z, self._vals[:, 1].flatten()))
                interpolant.append(Rbf(self._coords_x, self._coords_y, self._coords_z, self._vals[:, 2].flatten()))
            else:
                raise Exception("Problem dimension and data dimension not matching.")
        else:
            raise Exception("Dimension of the function invalid/not supported.")

        return interpolant

    def interpolate(self, x):
        assert ((self.is_scalar_valued() and self._vals.ndim == 1) or
                (self.is_vector_valued() and self._vals.ndim == self._dimension))

        return_value = self._vals.ndim * [None]

        if self._dimension == 1:
            for i in range(self._vals.ndim):
                return_value[i] = self._f[i](x[0])
        if self._dimension == 2:
            for i in range(self._vals.ndim):
                return_value[i] = self._f[i](x[0], x[1])
        if self._dimension == 3:
            for i in range(self._vals.ndim):
                return_value[i] = self._f[i](x[0], x[1], x[2])
        return return_value


class ExactInterpolationExpression(CustomExpression):
    """Uses cubic spline interpolation for implementation of CustomExpression.interpolate. Only allows interpolation on
    coupling that are parallel to the y axis, and if the coordinates in self._coords_y are ordered such that the nodes
    on the coupling mesh are traversed w.r.t their connectivity.
    However, this method allows to exactly recover the solution at the coupling interface, if it is a polynomial of
    order 3 or lower.
    See also https://github.com/precice/fenics-adapter/milestone/1
    """

    def create_interpolant(self):
        interpolant = []
        if self._dimension == 2:
            n_samples = len(self._coords_y)
            if n_samples > 3:
                kind = "cubic"
            else:
                logger.warning("Only {n_samples} nodes for interpolation provided. This may reduce the accuracy. "
                                "Consider refining your mesh.".format(n_samples=n_samples))
                if n_samples == 3:
                    kind = "quadratic"
                elif n_samples == 2:
                    kind = "linear"
                else:
                    raise Exception("Not sufficient number nodes for interpolation provided!")

            if self.is_scalar_valued():  # check if scalar or vector-valued
                interpolant.append(
                    interp1d(self._coords_y, self._vals, bounds_error=False, fill_value="extrapolate", kind=kind))
            elif self.is_vector_valued():
                interpolant.append(
                    interp1d(self._coords_y, self._vals[:, 0].flatten(), bounds_error=False, fill_value="extrapolate",
                             kind=kind))
                interpolant.append(
                    interp1d(self._coords_y, self._vals[:, 1].flatten(), bounds_error=False, fill_value="extrapolate",
                             kind=kind))
            else:
                raise Exception("Problem dimension and data dimension not matching.")
        else:
            raise Exception("Dimension of the function is invalid/not supported.")

        return interpolant

    def interpolate(self, x):
        assert ((self.is_scalar_valued() and self._vals.ndim == 1) or
                (self.is_vector_valued() and self._vals.ndim == self._dimension))

        return_value = self._vals.ndim * [None]

        if self._dimension == 2:
            for i in range(self._vals.ndim):
                return_value[i] = self._f[i](x[1])
        else:
            raise Exception("invalid dimensionality!")
        return return_value


class EmptyExpression(CustomExpression):
    """A dummy expression that can be used for implementing a coupling boundary condition, if the participant's mesh has
    no vertices on the coupling domain. Only used for parallel runs.

    Example:
    We want solve
    F = u * v / dt * dx + dot(grad(u), grad(v)) * dx - (u_n / dt + f) * v * dx + v * coupling_expression * ds
    The user defines F, but does not know whether the rank even has vertices on the Neumann coupling boundary.
    If the rank does not have any vertices on the Neumann coupling boundary the coupling_expression is an
    EmptyExpression. This "deactivates" the Neumann BC for that specific rank.
    """

    def eval(self, value, x):
        """ Evaluates expression at x. For EmptyExpression always returns zero.

        :param x: coordinate where expression has to be evaluated
        :param value: buffer where result has to be returned to
        """
        assert(MPI.size(MPI.comm_world) > 1)
        for i in range(self._vals.ndim):
            value[i] = 0

    def update_boundary_data(self, vals, coords_x, coords_y=None, coords_z=None):
        pass  # an EmptyExpression is never updated
