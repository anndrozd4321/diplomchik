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

import itertools
import collections

import numpy as np


class Patch:
    def __init__(self, xxyy_limits):
        """Specified by 'xxyy_limits' a sequence of two pairs: [[x_low,
        x_high], [y_low, y_high]]."""
        self.xlims, self.ylims = np.array(xxyy_limits)

    def __repr__(self):
        return 'Patch: x {} -> {} , y {} -> {}'.format(self.xlims[0], self.xlims[1], self.ylims[0], self.ylims[1])

    @property
    def side_lengths(self):
        return (self.xlims[1] - self.xlims[0], self.xlims[1] - self.xlims[0])
    
    @property
    def area(self):#
        xl, yl = self.xlims, self.ylims
        return (xl[1] - xl[0]) * (yl[1] - yl[0])

    @property
    def perimeter_length(self):#
        xl, yl = self.xlims, self.ylims
        return 2 * ((xl[1] - xl[0]) + (yl[1] - yl[0]))

    @property
    def corners(self):
        return np.array(list(itertools.product(self.xlims, self.ylims)))
    
    def intersect(self, other):
        """Compute the intersection with 'other'. If overlapping a Patch will
        be returned representing the overlap. Otherwise 'None' is
        returned.
        """
        xlims_new = max(self.xlims[0], other.xlims[0]), min(self.xlims[1], other.xlims[1])
        ylims_new = max(self.ylims[0], other.ylims[0]), min(self.ylims[1], other.ylims[1])
        if xlims_new[0] < xlims_new[1] and ylims_new[0] < ylims_new[1]:
            return Patch((xlims_new, ylims_new))
        else:
            return None

    def grown(self, growth):
        """Return a Patch which is this patch grown by the amount specified in
        'growth'. 'growth' may be a single number, for uniform growth,
        or a sequence of two numbers, for growth in the x- and
        y-directions respectively.
        """ 
        if isinstance(growth, collections.Sequence):
            gx, gy = growth
        else:
            gx = gy = growth
        xl, yl = self.xlims, self.ylims
        return Patch(((xl[0]-gx, xl[1]+gx), (yl[0]-gy, yl[1]+gy)))

def _test():
    p1=Patch(((1,2),(3,4)))
    p2=Patch(((1,1.5),(3.5,8)))
    pint = p1.intersect(p2)
    print(pint)
