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

from . import Vector, Orientation, Transform

def test_ref():
    """Test that the 'orient' and 'pos' properties of a Transform gets
    reference to the Transform data.
    """
    t = Transform(1,(2,3))
    o = t.orient
    o.angle = -1
    assert(t.orient.angle==-1)
    p = t.pos
    p.x = 0.5
    assert(t.pos.x == 0.5)

def test_mul():
    """Test multiplications, i.e. transformations, inner products, or
    group products.
    """
    o1 = Orientation(1.0)
    o2 = Orientation(-2.0)
    assert(isinstance(o1 * o2, Orientation))
    assert((o1 * o2).angle == -1.0)
    v1 = Vector((1, 2))
    v2 = Vector((-2, 1))
    v3 = Vector((0, 1))
    assert(v1 * v2 == 0.0)
    assert(v3 * v1 == v1[1])
    assert(v3 * v2 == v2[1])
    
if __name__ == '___main__':
    print('Testing')
    test_ref()
