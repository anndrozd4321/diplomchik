# coding=utf-8

"""
"""

__author__ = "Morten Lind"
__copyright__ = "Morten Lind 2016-2019"
__credits__ = ["Morten Lind"]
__license__ = "GPLv3"
__maintainer__ = "Morten Lind"
__email__ = "morten@lind.dyndns.dk"
__status__ = "Development"


import numpy as np
from .. import Vector, Transform
from .. import EPS
from .patch import Patch

class Line(object):
    """A mathematical line, i.e. double infinite."""

    def __init__(self, start, end):
        self._start = Vector(start)
        self._end = Vector(end)
        self._update()

    def __repr__(self):
        return 'Line(p{}, d{})'.format(self._start.tolist(),
                                       self._dir.tolist())

    def fit_points(self, points):
        """Compute the line from a set of points. 'points'
        must be an array of row position vectors, such that
        points[i] is a position vector."""
        points = np.array(points)
        centre = np.sum(points, axis=0)/len(points)
        eigen = np.linalg.eig(np.cov(points.T))
        max_ev_i = np.where(eigen[0] == max(eigen[0]))[0][0]
        direction = eigen[1].T[max_ev_i]
        self._start = Vector(centre)
        self._dir = Vector(direction)
        self._end = self._start + self._dir

    @classmethod
    def new_fitted_points(cls, points):
        obj = cls(1, 2)
        obj.fit_points(points)
        return obj

    @classmethod
    def new_point_dir(cls, point, direction):
        obj = cls(point, point+direction)
        return obj

    @classmethod
    def new_from_coeffs(cls, a, b):
        start = Vector(0, b)
        end = Vector(1, b + a)
        return cls(start, end)

    @property
    def dir(self):
        """Get the direction of the line."""
        return self._dir.copy()

    @property
    def point(self):
        """Get a point on the line"""
        return self._start.copy()

    @property
    def unit_direction(self):
        return self._dir.copy()

    def _update(self):
        """Update derived data from changed fundamental data."""
        self._dir = (self._end - self._start).normalized

    def reverse(self):
        """Simply reverses the unit direction vector and swaps the start and
        end points.
        """
        start = self._end
        end = self._start
        self._start = Vector(start)
        self._end = Vector(end)
        self._update()

    @property
    def reversed(self):
        return LineSegment(self._end, self._start)

    
    def transform(self, trf):
        """Apply the given transform inline."""
        self._start = trf * self._start
        self._end = trf * self._end
        Line._update(self)

    def __rmul__(self, trf):
        """Return the homogeneous transformed of this line."""
        if type(trf) == Transform:
            return Line(trf * self._start, trf * (self._start + self._dir))
        else:
            return NotImplemented

    def project(self, pos):
        """Return the projection of p onto the line"""
        return self._start + (pos - self._start) * self._dir * self._dir

    def __contains__(self, pos):
        """Test if 'pos' is on the line segment."""
        # Shortest dist vector must be of 0 length
        return (self.project(pos) - pos).norm < EPS

    def _l_intersect(self, ls):
        """http://stackoverflow.com/a/565282"""
        s = self
        o = ls
        s2o = o._start - s._start
        dcross = np.cross(s._dir, o._dir)
        if np.abs(dcross) < EPS:
            return []
        else:
            # The paramerers t for which the line intersection problem
            # matches the lines:
            st = np.cross(s2o, o.dir) / dcross
            return [s._start + st * s._dir]

    def intersection(self, other):
        if type(other) == Line:
            return self._l_intersect(other)
        else:
            return other.intersection(self)

    def dist(self, other):
        """Compute the distance to 'other'."""
        if type(other) == Vector:
            pp = self.project(other)
            return other.dist(pp)
        else:
            raise NotImplementedError

    @property
    def coeffs(self):
        """Return the coefficients, (inclination, intercept) for this line."""
        a = self._dir.y / self._dir.x
        b = self._start.y - self._start.x * a
        return (a, b)


class LineSegment(Line):
    """A segment of a line."""

    def __init__(self, start, end):
        Line.__init__(self, start, end)
        self._update()

    def _update(self):
        """Update derived data from changed fundamental data."""
        self._delta = self._end - self._start
        self._dir = self._delta.normalized
        self.length = self._delta.norm
        self.centre = self._start + 0.5 * self._delta

    @property
    def dir(self):
        """Get the direction of the line segment."""
        return self._dir.copy()

    @property
    def delta(self):
        """Get the segment vector of the line segment."""
        return self._delta.copy()

    @property
    def start(self):
        """Get the start of the line segment."""
        return self._start.copy()

    @property
    def end(self):
        """Get the end of the line segment."""
        return self._end.copy()

    @property
    def line(self):
        """Get an instance of the line, on which this segment lies."""
        return Line(self._start, self._end)

    @classmethod
    def new_from_coeffs(cls, a, b, x_start, x_end):
        start = Vector(x_start, b + a * x_start)
        end = Vector(x_end, b + a * x_end)
        return LineSegment(start, end)

    @classmethod
    def new_from_dxf(cls, dxf_line):
        return LineSegment(dxf_line.start, dxf_line.end)

    @classmethod
    def new_fitted_points(cls, points):
        """Fit the given points to a line, and set start and end according to
        the min and max projection of the points. 'points' must be a
        2D iterable of row vectors, i.e. shape == (n, 2).
        """
        points = np.array(points)
        line = Line.new_fitted_points(points)
        projs = (points - line.point) * line.unit_direction
        start = line.point + projs.min() * line.unit_direction
        end = line.point + projs.max() * line.unit_direction
        return LineSegment(start, end)

    def __contains__(self, p):
        """Test if 'p' is on the line segment."""
        pp = self.project(p)
        # Check if projection distance is different from 0, to some
        # level of precision
        if (pp - p).norm > EPS:
            return False
        pp_along = (pp - self._start) * self._dir / self.length
        if pp_along <= 1 and pp_along >= 0:
            return True
        else:
            return False

    def transform(self, trf):
        """Apply the given transform inline."""
        Line.transform(self, trf)
        LineSegment._update(self)

    def __rmul__(self, trf):
        """Return the homogeneously transformed of this line segment."""
        if type(trf) == Transform:
            return LineSegment(trf * self._start, trf * self._end)
        else:
            return NotImplemented

    def __repr__(self):
        return 'LineSegment({} -> {})'.format(self._start.tolist(),
                                              self._end.tolist())

    def dist(self, other):
        """Compute the distance to 'other'."""
        if type(other) == Vector:
            pp = self.project(other)
            if pp in self:
                return other.dist(pp)
            else:
                return min(other.dist(self._start), other.dist(self._end))
        else:
            raise NotImplementedError

    def _ls_intersect(self, ls):
        """http://stackoverflow.com/a/565282"""
        s = self
        o = ls
        s2o = o._start - s._start
        dcross = np.cross(s._delta, o._delta)
        if np.abs(dcross) < EPS:
            return []
        # The paramerers t for which the line intersection problem
        # matches the lines:
        st = np.cross(s2o, o._delta) / dcross
        if st < 0 or st > 1:
            return []
        ot = np.cross(s2o, s._delta) / dcross
        if ot >= 0 and ot <= 1:
            return [s._start + st * s._delta]
        else:
            return []

    def _l_intersect(self, l):
        """Based on http://stackoverflow.com/a/565282"""
        s = self
        o = l
        s2o = o._start - s._start
        dcross = np.cross(s._delta, o.dir)
        if np.abs(dcross) < EPS:
            return []
        # The paramerers t for which the line intersection problem
        # matches the lines:
        st = np.cross(s2o, o.dir) / dcross
        if st < 0 or st > 1:
            return []
        else:
            return [s._start + st * s._delta]

    def intersection(self, other):
        if type(other) == LineSegment:
            return self._ls_intersect(other)
        elif type(other) == Line:
            return self._l_intersect(other)
        else:
            return other.intersection(self)

    @property
    def coeffs(self):
        a, b = Line.coeffs.fget(self)
        return (a, b, self.start.x, self.end.x)


class Circle(object):
    def __init__(self, centre, radius):
        self.centre = Vector(centre)
        self.radius = radius

    def transform(self, trf):
        """Apply the given transform inline."""
        self.centre = trf * self.centre

    def __rmul__(self, trf):
        """Return the homogeneously transformed of this circle."""
        if type(trf) == Transform:
            return Circle(trf * self.centre, self.radius)
        else:
            return NotImplemented

    def __repr__(self):
        return 'Circle(c{}, r[{}])'.format(self.centre.tolist(), self.radius)

    def _l_intersection(self, line):
        cp = line.project(self.centre)
        cp_dist = (cp - self.centre).norm
        xes = []
        if cp_dist < self.radius:
            # Two possibles: The half chords form triangles with a
            # radius and closest point vector
            hcl = np.sqrt(self.radius**2 - cp_dist**2)
            # This must be added and subtracted to the closes point
            # along the segment direction
            hc = hcl * line.dir
            xes = [cp + hc, cp - hc]
        elif cp_dist == self.radius:
            # one possible
            xes = [cp]
        return xes

    def __contains__(self, p):
        """Compute if p is on the circle."""
        return (p - self.centre).norm - self.radius < EPS

    def intersection(self, obj):
        if type(obj) == Line:
            return Circle._l_intersection(self, obj)
        else:
            return obj.intersection(self)


class CircleSegment(Circle):
    def __init__(self, centre, radius, a_start, a_end):
        Circle.__init__(self, centre, radius)
        self.a0 = np.deg2rad(a_start)
        self.a1 = np.deg2rad(a_end)
        CircleSegment._update(self)

    def _update(self):
        """Update derived data from changed fundamental data."""
        self._start = self.centre + self.radius * Vector([np.cos(self.a0),
                                                         np.sin(self.a0)])
        self._end = self.centre + self.radius * Vector([np.cos(self.a1),
                                                       np.sin(self.a1)])
        self.chord = self._end - self._start

    def transform(self, trf):
        """Apply the given transform inline."""
        Circle.transform(self, trf)
        tang = trf.orient.angle
        self.a0 = (self.a0 + tang) % (2 * np.pi)
        self.a1 = (self.a1 + tang) % (2 * np.pi)
        CircleSegment._update(self)

    @classmethod
    def new_from_dxf(cls, dxf_arc):
        """Create a CircleSegment from a DXF Arc."""
        darc = dxf_arc
        return CircleSegment(darc.center, darc.radius,
                             darc.start_angle, darc.end_angle)

    def __contains__(self, p):
        """Evaluate if p is on the circle segment."""
        return (Circle.__contains__(self, p)
                and np.cross(self.chord, p-self._start) <= 0)

    def __rmul__(self, trf):
        """Return the homogeneously transformed of this circle segment."""
        if type(trf) == Transform:
            tang = trf.orient.angle
            a0 = (self.a0 + tang) % (2 * np.pi)
            a1 = (self.a1 + tang) % (2 * np.pi)
            return CircleSegment(trf * self.centre, self.radius, a0, a1)
        else:
            return NotImplemented

    def __repr__(self):
        return 'Circle(c{}, r[{}])'.format(self.centre.tolist(), self.radius)

    def _ls_intersection(self, ls):
        # First find possible line-circle collisions
        xposs = Circle.intersection(self, ls)
        # Filter points that are not on the circle segment
        return [x for x in xposs if x in self and x in ls]

    def _l_intersection(self, l):
        # First find possible line-circle collisions
        xposs = Circle.intersection(self, l)
        # Filter points that are not on the circle segment
        return [x for x in xposs if x in self]

    def intersection(self, other):
        if type(other) == LineSegment:
            return self._ls_intersection(other)
        elif type(other) == Line:
            return self._l_intersection(other)
        else:
            return other.intersection(self)


def _test():
    """Some (incomplete) testing."""
    # Testing Line creation and intersection
    l0 = Line((1, 2), (2, 4))
    l1 = Line((1, 0), (0, 2))
    assert((l1.intersection(l0)[0] -
            Vector(0.5, 1)).norm == 0.0)
    # Testing LineSegment creation and intersection
    ls0 = LineSegment((1, 2), (2, 4))
    ls1 = LineSegment((1, 0), (0, 2))
    assert len(ls1.intersection(ls0)) == 0
    ls2 = LineSegment((0, 0), (1, 2))
    assert (ls1.intersection(ls2)[0] -
            Vector(0.5, 1)).norm == 0.0
    # Testing line fitting
    points = [[2, 2],
              [1, 1],
              [3, 3]]
    fl = Line.new_fitted_points(points)
    assert(fl.point == Vector(2, 2))
    assert(1.0 - np.abs(fl.dir * Vector(1, 1).normalized)
           < 10 * EPS)
    fls = LineSegment.new_fitted_points(points)
    assert(fls.point == Vector(1, 1))
    assert(abs(1.0 - np.abs(fls.dir * Vector(1, 1).normalized))
               < 10 * EPS)
    assert(abs(fls.length - 2 * np.sqrt(2)) < 10 * EPS)
    # Testing LineSegment coefficients property
    ls = LineSegment.new_from_coeffs(1, 2, 1, 2)
    assert (ls.start.x == 1 and ls.start.y == 3 and
            ls.end.x == 2 and ls.end.y == 4)
    assert ls.coeffs[0] == 1 and ls.coeffs[1] == 2
    # Test reversal of Line and LineSegment
    # ...
