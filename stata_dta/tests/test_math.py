from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

from ..compatibility import range, zip, builtin

import math

import numpy as np
from numpy import linspace
from nose.tools import assert_almost_equals
from nose.plugins.skip import SkipTest

from .util import raises
from .. import Dta115, Dta117
from ..stata_missing import MISSING_VALS as mvs, MISSING as mv
from .. import stata_math as st_math
    

# number of decimal places to check for float equality
NUM_PLACES = 10

def cloglog(x):
    log = math.log
    return log(-log(1 - x))


def _get_listified_domain(domain):
    return [[i] for i in domain]


def _test_function(dta, domain, m_func, st_func):
    av = st_func(dta.var0_)[:len(domain)]
    ev = map(m_func, domain)

    for a, e in zip(av, ev):
        if not (a == mv or round(a-e, NUM_PLACES) == 0):
            return False
    return True


def _get_math_function(name):
    fctn = getattr(math, name, None)
    if fctn is None:
        fctn = getattr(builtin, name, None)
        if fctn is None:
            fctn = globals()[name]
    return fctn


class TestMissingVars(object):

    def test_missing_math1(self):
        assert mvs[0] * mvs[1] == mv

    def test_missing_math2(self):
        assert mvs[10] + 123.456 == mv

    def test_missing_math3(self):
        assert raises(math.sin, (TypeError, AttributeError), mvs[0])

    def test_missing_math4(self):
        data = Dta117([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        assert raises(math.sin, TypeError, data.var0_)


# maps the function to test to its domain
math_fctns = {'acos': linspace(-1, 1, 1000),
              'abs': linspace(-100, 100, 1000),
              'acosh': linspace(1, 100, 1000),
              'asin': linspace(-1, 1, 1000),
              'asinh': linspace(-100, 100, 1000),
              'atan': linspace(-100, 100, 1000),
              'cos': linspace(-100, 100, 1000),
              'cosh': linspace(-100, 100, 1000),
              'exp': linspace(-100, 100, 1000),
              }


def test_fctns_mv():
    for name in math_fctns.keys():
        for missing_value in mvs:
            assert getattr(st_math, 'st_%s' % name)(missing_value) == mv, \
                "st_%s fails to handle missing values" % name


def test_math_fctns_StataVariable():
    for name, domain in math_fctns.items():
        m_func = _get_math_function(name)
        st_func = getattr(st_math, "st_%s" % name)

        dta117 = Dta117(_get_listified_domain(domain))
        assert _test_function(dta117, domain, m_func, st_func), \
            "st_%s fails to handle StataVariables" % name


def test_math_fctns_StataVariable_and_mv():
    raise SkipTest
    for name, domain in math_fctns.items():
        np.insert(np.copy(domain), 1, domain[0])
        m_func = _get_math_function(name)
        st_func = getattr(st_math, "st_%s" % name)

        dta117 = Dta117(
            _get_listified_domain(np.insert(domain.astype(object), 1, mv)))
        assert _test_function(dta117, domain, m_func, st_func), \
            "st_%s fails to handle missing values with StataVariables" % name


# These are functions that the naive approach fails on.  More work will have to 
# be done to test them

#atan2

#atanh

#ceil

#cloglog

#comb

#digamma