from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

from math import sin

from .util import raises
from .. import Dta117
from ..stata_missing import MISSING_VALS as mvs, MISSING as mv
from ..stata_math import *
    

def test_missing_math1():
	assert mvs[0] * mvs[1] == mv


def test_missing_math2():
	assert mvs[10] + 123.456 == mv

def test_missing_math3():
	assert raises(sin, (TypeError, AttributeError), mvs[0])

def test_missing_math4():
	data = Dta117([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
	assert raises(sin, TypeError, data.var0_)
