"""
This module provides a mechanism to imterpolate point data acquired from preCICE into FEniCS Expressions.
"""

import dolfin
from dolfin import UserExpression
from scipy.interpolate import Rbf
from scipy.interpolate import interp1d
import numpy as np

import logging

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class CustomExpression(UserExpression):
    """
    Creates functional representation (for FEniCS) of nodal data provided by preCICE.
    """

    def update_boundary_data(self, vals, coords_x, coords_y=None, coords_z=None):
        """
        Update object of this class of type FEniCS UserExpression with given point data.

        Parameters
        ----------
        vals : double
            Point data to be used to update the Expression.
        coords_x : double
            X coordinate of points of which point data is provided.
        coords_y : double
            Y coordinate of points of which point data is provided.
        coords_z : double
            Z coordinate of points of which point data is provided.
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
        # TODO: the correct way to deal with this would be using an abstract class. Since this is technically more
        #  complex and the current implementation is a workaround anyway, we do not use the proper solution, but this hack.
        """
        Interpolates at x. Uses buffered interpolant self._f.
        Parameters
        ----------
        x : double
            Point.

        Returns
        -------
        list : python list
            A list containing the interpolated values. If scalar function is interpolated this list has a single
            element. If a vector function is interpolated the list has self._dimensions elements.
        """
        raise Exception("Please use one of the classes derived from this class, that implements an actual strategy for"
                        "interpolation.")

    def create_interpolant(self, x):
        # TODO: the correct way to deal with this would be using an abstract class. Since this is technically more
        #  complex and the current implementation is a workaround anyway, we do not use the proper solution, but this hack.
        """
        Creates interpolant from boundary data that has been provided before.

        Parameters
        ----------
        x : double
            Point.

        Returns
        -------
        list : python list
            Interpolant as a list. If scalar function is interpolated this list has a single
            element. If a vector function is interpolated the list has self._dimensions elements.
        """
        raise Exception("Please use one of the classes derived from this class, that implements an actual strategy for"
                        "interpolation.")

    def eval(self, value, x):
        """
        Evaluates expression at x using self.interpolate(x) and stores result to value.

        Parameters
        ----------
        value : double
            Buffer where result has to be returned to.
        x : double
            Coordinate where expression has to be evaluated.
        """
        return_value = self.interpolate(x)
        for i in range(self._vals.ndim):
            value[i] = return_value[i]

    def is_scalar_valued(self):
        """
        Determines if function being interpolated is scalar-valued based on dimension of provided vector self._vals.

        Returns
        -------
        tag : bool
            True if function being interpolated is scalar-valued, False otherwise.
        """
        if self._vals.ndim == 1:
            return True
        elif self._vals.ndim > 1:
            return False
        else:
            raise Exception("Dimension of the function is 0 or negative!")

    def is_vector_valued(self):
        """
        Determines if function being interpolated is vector-valued based on dimension of provided vector self._vals.

        Returns
        -------
        tag : bool
            True if function being interpolated is vector-valued, False otherwise.
        """
        if self._vals.ndim > 1:
            return True
        elif self._vals.ndim == 1:
            return False
        else:
            raise Exception("Dimension of the function is 0 or negative!")


class GeneralInterpolationExpression(CustomExpression):
    """
    Uses RBF interpolation for implementation of CustomExpression.interpolate. Allows for arbitrary coupling
    interfaces, but has limited accuracy.
    """

    def create_interpolant(self):
        """
        See base class description.
        """
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
        """
        See base class description.
        """
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
    """
    Uses cubic spline interpolation for implementation of CustomExpression.interpolate. Only allows interpolation on
    coupling that are parallel to the y axis, and if the coordinates in self._coords_y are ordered such that the nodes
    on the coupling mesh are traversed w.r.t their connectivity.
    However, this method allows to exactly recover the solution at the coupling interface, if it is a polynomial of
    order 3 or lower.
    See also https://github.com/precice/fenics-adapter/milestone/1
    """

    def create_interpolant(self):
        """
        See base class description.
        """
        interpolant = []
        if self._dimension == 2:
            if self.is_scalar_valued():  # check if scalar or vector-valued
                interpolant.append(
                    interp1d(self._coords_y, self._vals, bounds_error=False, fill_value="extrapolate", kind="cubic"))
            elif self.is_vector_valued():
                interpolant.append(
                    interp1d(self._coords_y, self._vals[:, 0].flatten(), bounds_error=False, fill_value="extrapolate",
                             kind="cubic"))
                interpolant.append(
                    interp1d(self._coords_y, self._vals[:, 1].flatten(), bounds_error=False, fill_value="extrapolate",
                             kind="cubic"))
            else:
                raise Exception("Problem dimension and data dimension not matching.")
        else:
            raise Exception("Dimension of the function is invalid/not supported.")

        return interpolant

    def interpolate(self, x):
        """
        See base class description.
        """
        assert ((self.is_scalar_valued() and self._vals.ndim == 1) or
                (self.is_vector_valued() and self._vals.ndim == self._dimension))

        return_value = self._vals.ndim * [None]

        if self._dimension == 2:
            for i in range(self._vals.ndim):
                return_value[i] = self._f[i](x[1])
        else:
            raise Exception("invalid dimensionality!")
        return return_value
