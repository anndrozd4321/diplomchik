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


import numpy as np

# Level of precision
EPS = 1000 * np.finfo(np.float).eps

from .vector import Vector
from .orientation import Orientation
from .transform import Transform
from . import geometry

