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

import numpy as np

from .vector import Vector
from .orientation import Orientation


class Transform(np.ndarray):
    def __new__(cls, *args, **kwargs):
        return np.ndarray.__new__(cls, (3, 3), dtype=np.float)

    def __init__(self, *args):
        self[:, :] = np.identity(3)
        if len(args) == 2:
            if isinstance(args[1], Iterable):
                # Argument at index 1 must be a vector
                self[:2, :2] = Orientation(args[0])
                self[:2, 2] = args[1]
            else:
                raise NotImplementedError(
                    'Argument two must be iterable and form a translation')
        elif len(args) == 1:
            if isinstance(args[0], Transform):
                self[:, :] = args[0][:, :]
            else:
                raise NotImplementedError(
                    'Single argument must be a Transform object')
        elif len(args) == 0:
            # Default constructor of identity orientation and zero
            # vector. These are fulfilled by default.
            # self[:2, :2] = np.identity(2)
            # self[:2, 2] = 0
            pass
        else:
            raise NotImplementedError('Need zero, one, or two arguments.')

    def __array_finalize__(self, obj):
        if obj is None:
            return

    def __array_wrap__(self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self, out_arr, context)

    def __mul__(self, other):
        if isinstance(other, Vector):
            return Vector(self[:2, :2].dot(other) + self[:2, 2])
        elif isinstance(other, Transform):
            return self.dot(other)
        else:
            return NotImplemented

    @property
    def array_ref(self):
        return self.view(np.ndarray)

    @property
    def array(self):
        return np.array(self, copy=True)

    def __getitem__(self, slice):
        return np.array(self, copy=False)[slice]

    def get_orient(self):
        """Return a reference to the orientation part of the transform.
        """
        return self[:2, :2].view(Orientation)

    def set_orient(self, new_orient):
        """Copy the orientation data from 'new_orient'.
        """
        self[:2, :2] = new_orient

    orient = property(get_orient, set_orient)

    def get_pos(self):
        return self[:2, 2].view(Vector)

    def set_pos(self, new_pos):
        self[:2, 2] = new_pos

    pos = property(get_pos, set_pos)

    @property
    def inverse(self):
        return np.linalg.inv(self)

    def invert(self):
        self[:, :] = np.linalg.inv(self)


def _test():
    t = Transform(1, (2, 3))
    print(t.orient)
    print(t.pos)
    print('Test inverse: ', np.allclose(t*t.inverse, np.identity(3)))
