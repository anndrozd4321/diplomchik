# coding=utf-8

"""
"""

__author__ = "Morten Lind"
__copyright__ = "Morten Lind 2016-2017"
__credits__ = ["Morten Lind"]
__license__ = "GPLv3"
__maintainer__ = "Morten Lind"
__email__ = "morten@lind.dyndns.dk"
__status__ = "Development"


from collections import Iterable
from numbers import Number

import numpy as np

from .vector import Vector
import math2d


class Orientation(np.ndarray):

    def __new__(cls, *args, **kwargs):
        return np.ndarray.__new__(cls, (2, 2), dtype=np.float)

    def __init__(self, *args, **kwargs):
        if len(args) == 0:
            self[:, :] = np.identity(2)
        elif len(args) == 1:
            if isinstance(args[0], Iterable):
                self[:, :] = np.array(args[0], copy=False)
            else:
                self.angle = args[0]
        else:
            raise NotImplementedError('Need zero or one argument.')

    def __array_finalize__(self, obj):
        if obj is None:
            return

    def __array_wrap__(self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self, out_arr, context)

    def __mul__(self, other):
        """Rotation of a Vector or group product in SO(2)"""
        if isinstance(other, Vector):
            return self.dot(other).view(Vector)
        elif isinstance(other, Orientation):
            return self.dot(other).view(Orientation)
        else:
            return NotImplemented

    @property
    def array_ref(self):
        """Return a reference to the array data."""
        return self.view(np.nparray)

    @property
    def array(self):
        return np.array(self, copy=True)

    def __getitem__(self, slc):
        return np.array(self, copy=False)[slc]

    # @classmethod
    # def new_by_ref(cls, data):
    #     """Create a new Orientation object with reference to 'data'."""
    #     o = Orientation()
    #     if isinstance(data, Orientation):
    #         pass

    def get_vec_x(self):
        """Return the x-direction of the moving coordinate system in
        base reference as a Vector.
        """
        return Vector(self[:, 0])

    vec_x = property(get_vec_x)

    def get_col_x(self):
        """Return the x-direction of the moving coordinate system in
        base reference as an array."""
        return self[:, 0]

    col_x = property(get_col_x)

    def get_vec_y(self):
        """Return the y-direction of the moving coordinate system in
        base reference as a Vector.
        """
        return Vector(self[:, 1])

    vec_y = property(get_vec_y)

    def get_angle_02pi(self):
        """Return the angle in [0;2pi) represented by this orientation."""
        return np.arctan2(*self[::-1, 0]) % (2 * np.pi)
    angle02pi = property(get_angle_02pi)

    def get_angle(self):
        """Return the angle in (-pi;pi] represented by this orientation."""
        return np.arctan2(*self[::-1, 0])
    def set_angle(self, a):
        self[:, :] = [[np.cos(a), -np.sin(a)],
                      [np.sin(a), np.cos(a)]]
    angle = property(get_angle, set_angle)

    @property
    def inverse(self):
        return self.T

    def invert(self):
        self[:, :] = self.T
