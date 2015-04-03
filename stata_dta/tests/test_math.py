from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

from ..compatibility import range, zip, builtin, map

import math

import numpy as np
from numpy import linspace
from nose.tools import assert_almost_equals
from nose.plugins.skip import SkipTest

from .math_util import *
from .util import raises
from .. import Dta115, Dta117
from ..stata_missing import MISSING_VALS as mvs, MISSING as mv
from .. import stata_math

# number of decimal places to check for float equality
NUM_PLACES = 10


def _get_listified_domain(domain):
    return [[i] for i in domain]


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


class TestAbs(object):

    def test_missing_value(self):
        assert stata_math.st_abs(mv) == mv
        for mv_ in mvs:
            assert stata_math.st_abs(mv_) == mv

    def test_none(self):
        assert stata_math.st_abs(None) == mv

    def test_not_number(self):
        assert raises(stata_math.st_abs, TypeError, '')
        assert raises(stata_math.st_abs, TypeError, [])
        assert raises(stata_math.st_abs, TypeError, ())
        assert raises(stata_math.st_abs, TypeError, {})

    def test_invalid_number(self):
        x = -8.988465674311579e+307
        assert stata_math.st_abs(x) == abs(x)
        x -= 1e307
        assert stata_math.st_abs(x) == mv

        x = 8.988465674311579e+307
        assert stata_math.st_abs(x) == x
        x += 1e307
        assert stata_math.st_abs(x) == mv

    def test_valid_number(self):
        x = -1
        assert stata_math.st_abs(x) == 1
        x = 0
        assert stata_math.st_abs(x) == 0
        x = 1
        assert stata_math.st_abs(x) == 1


class Test_acos(object):
    domain = linspace(-1, 1, 3000)

    def test_missing_values(self):
        for mv_ in mvs:
            assert stata_math.st_acos(mv_) == mv

    def test_stata_variable(self):
        dta = Dta117(_get_listified_domain(self.domain))

        actual_values = stata_math.st_acos(dta.var0_)
        expected_values = list(map(_get_math_function('acos'), self.domain))

        for av, ev in zip(actual_values, expected_values):
            assert av == ev, \
                "st_acos fails to calculate stata variable values"

    def test_normal_values(self):
        actual_values = map(stata_math.st_acos, self.domain)
        expected_values = list(map(_get_math_function('acos'), self.domain))

        for av, ev in zip(actual_values, expected_values):
            assert av == ev, \
                "st_acos fails to calculate normal values"


class Test_abs(object):
    domain = linspace(-100, 100, 3000)

    def test_missing_values(self):
        for mv_ in mvs:
            assert stata_math.st_abs(mv_) == mv

    def test_stata_variable(self):
        dta = Dta117(_get_listified_domain(self.domain))

        actual_values = stata_math.st_abs(dta.var0_)
        expected_values = list(map(_get_math_function('abs'), self.domain))

        for av, ev in zip(actual_values, expected_values):
            assert av == ev, \
                "st_abs fails to calculate stata variable values"

    def test_normal_values(self):
        actual_values = map(stata_math.st_abs, self.domain)
        expected_values = list(map(_get_math_function('abs'), self.domain))

        for av, ev in zip(actual_values, expected_values):
            assert av == ev, \
                "st_abs fails to calculate normal values"


class Test_acosh(object):
    domain = linspace(1, 100, 3000)

    def test_missing_values(self):
        for mv_ in mvs:
            assert stata_math.st_acosh(mv_) == mv

    def test_stata_variable(self):
        dta = Dta117(_get_listified_domain(self.domain))

        actual_values = stata_math.st_acosh(dta.var0_)
        expected_values = list(map(_get_math_function('acosh'), self.domain))

        for av, ev in zip(actual_values, expected_values):
            assert av == ev, \
                "st_acosh fails to calculate stata variable values"

    def test_normal_values(self):
        actual_values = map(stata_math.st_acosh, self.domain)
        expected_values = list(map(_get_math_function('acosh'), self.domain))

        for av, ev in zip(actual_values, expected_values):
            assert av == ev, \
                "st_acosh fails to calculate normal values"


class Test_asin(object):
    domain = linspace(-1, 1, 3000)

    def test_missing_values(self):
        for mv_ in mvs:
            assert stata_math.st_asin(mv_) == mv

    def test_stata_variable(self):
        dta = Dta117(_get_listified_domain(self.domain))

        actual_values = stata_math.st_asin(dta.var0_)
        expected_values = list(map(_get_math_function('asin'), self.domain))

        for av, ev in zip(actual_values, expected_values):
            assert av == ev, \
                "st_asin fails to calculate stata variable values"

    def test_normal_values(self):
        actual_values = map(stata_math.st_asin, self.domain)
        expected_values = list(map(_get_math_function('asin'), self.domain))

        for av, ev in zip(actual_values, expected_values):
            assert av == ev, \
                "st_asin fails to calculate normal values"


class Test_asinh(object):
    domain = linspace(-100, 100, 3000)

    def test_missing_values(self):
        for mv_ in mvs:
            assert stata_math.st_asinh(mv_) == mv

    def test_stata_variable(self):
        dta = Dta117(_get_listified_domain(self.domain))

        actual_values = stata_math.st_asinh(dta.var0_)
        expected_values = list(map(_get_math_function('asinh'), self.domain))

        for av, ev in zip(actual_values, expected_values):
            assert av == ev, \
                "st_asinh fails to calculate stata variable values"

    def test_normal_values(self):
        actual_values = map(stata_math.st_asinh, self.domain)
        expected_values = list(map(_get_math_function('asinh'), self.domain))

        for av, ev in zip(actual_values, expected_values):
            assert av == ev, \
                "st_asinh fails to calculate normal values"


class Test_atan(object):
    domain = linspace(-100, 100, 3000)

    def test_missing_values(self):
        for mv_ in mvs:
            assert stata_math.st_atan(mv_) == mv

    def test_stata_variable(self):
        dta = Dta117(_get_listified_domain(self.domain))

        actual_values = stata_math.st_atan(dta.var0_)
        expected_values = list(map(_get_math_function('atan'), self.domain))

        for av, ev in zip(actual_values, expected_values):
            assert av == ev, \
                "st_atan fails to calculate stata variable values"

    def test_normal_values(self):
        actual_values = map(stata_math.st_atan, self.domain)
        expected_values = list(map(_get_math_function('atan'), self.domain))

        for av, ev in zip(actual_values, expected_values):
            assert av == ev, \
                "st_atan fails to calculate normal values"


class Test_cos(object):
    domain = linspace(-100, 100, 3000)

    def test_missing_values(self):
        for mv_ in mvs:
            assert stata_math.st_cos(mv_) == mv

    def test_stata_variable(self):
        dta = Dta117(_get_listified_domain(self.domain))

        actual_values = stata_math.st_cos(dta.var0_)
        expected_values = list(map(_get_math_function('cos'), self.domain))

        for av, ev in zip(actual_values, expected_values):
            assert av == ev, \
                "st_cos fails to calculate stata variable values"

    def test_normal_values(self):
        actual_values = map(stata_math.st_cos, self.domain)
        expected_values = list(map(_get_math_function('cos'), self.domain))

        for av, ev in zip(actual_values, expected_values):
            assert av == ev, \
                "st_cos fails to calculate normal values"


class Test_cosh(object):
    domain = linspace(-100, 100, 3000)

    def test_missing_values(self):
        for mv_ in mvs:
            assert stata_math.st_cosh(mv_) == mv

    def test_stata_variable(self):
        dta = Dta117(_get_listified_domain(self.domain))

        actual_values = stata_math.st_cosh(dta.var0_)
        expected_values = list(map(_get_math_function('cosh'), self.domain))

        for av, ev in zip(actual_values, expected_values):
            assert av == ev, \
                "st_cosh fails to calculate stata variable values"

    def test_normal_values(self):
        actual_values = map(stata_math.st_cosh, self.domain)
        expected_values = list(map(_get_math_function('cosh'), self.domain))

        for av, ev in zip(actual_values, expected_values):
            assert av == ev, \
                "st_cosh fails to calculate normal values"


class Test_exp(object):
    domain = linspace(-100, 100, 3000)

    def test_missing_values(self):
        for mv_ in mvs:
            assert stata_math.st_exp(mv_) == mv

    def test_stata_variable(self):
        dta = Dta117(_get_listified_domain(self.domain))

        actual_values = stata_math.st_exp(dta.var0_)
        expected_values = list(map(_get_math_function('exp'), self.domain))

        for av, ev in zip(actual_values, expected_values):
            assert av == ev, \
                "st_exp fails to calculate stata variable values"

    def test_normal_values(self):
        actual_values = map(stata_math.st_exp, self.domain)
        expected_values = list(map(_get_math_function('exp'), self.domain))

        for av, ev in zip(actual_values, expected_values):
            assert av == ev, \
                "st_exp fails to calculate normal values"


class Test_invcloglog(object):
    domain = linspace(-100, 100, 3000)

    def test_missing_values(self):
        for mv_ in mvs:
            assert stata_math.st_invcloglog(mv_) == mv

    def test_stata_variable(self):
        dta = Dta117(_get_listified_domain(self.domain))

        actual_values = stata_math.st_invcloglog(dta.var0_)
        expected_values = list(map(_get_math_function('invcloglog'), self.domain))

        for av, ev in zip(actual_values, expected_values):
            assert av == ev, \
                "st_invcloglog fails to calculate stata variable values"

    def test_normal_values(self):
        actual_values = map(stata_math.st_invcloglog, self.domain)
        expected_values = list(map(_get_math_function('invcloglog'), self.domain))

        for av, ev in zip(actual_values, expected_values):
            assert av == ev, \
                "st_invcloglog fails to calculate normal values"


class Test_invlogit(object):
    domain = linspace(-100, 100, 3000)

    def test_missing_values(self):
        for mv_ in mvs:
            assert stata_math.st_invlogit(mv_) == mv

    def test_stata_variable(self):
        dta = Dta117(_get_listified_domain(self.domain))

        actual_values = stata_math.st_invlogit(dta.var0_)
        expected_values = list(map(_get_math_function('invlogit'), self.domain))

        for av, ev in zip(actual_values, expected_values):
            assert av == ev, \
                "st_invlogit fails to calculate stata variable values"

    def test_normal_values(self):
        actual_values = map(stata_math.st_invlogit, self.domain)
        expected_values = list(map(_get_math_function('invlogit'), self.domain))

        for av, ev in zip(actual_values, expected_values):
            assert av == ev, \
                "st_invlogit fails to calculate normal values"


class Test_ln(object):
    domain = linspace(1, 100, 3000)

    def test_missing_values(self):
        for mv_ in mvs:
            assert stata_math.st_ln(mv_) == mv

    def test_stata_variable(self):
        dta = Dta117(_get_listified_domain(self.domain))

        actual_values = stata_math.st_ln(dta.var0_)
        expected_values = list(map(_get_math_function('ln'), self.domain))

        for av, ev in zip(actual_values, expected_values):
            assert av == ev, \
                "st_ln fails to calculate stata variable values"

    def test_normal_values(self):
        actual_values = map(stata_math.st_ln, self.domain)
        expected_values = list(map(_get_math_function('ln'), self.domain))

        for av, ev in zip(actual_values, expected_values):
            assert av == ev, \
                "st_ln fails to calculate normal values"


class Test_lngamma(object):
    domain = linspace(1, 100, 3000)

    def test_missing_values(self):
        for mv_ in mvs:
            assert stata_math.st_lngamma(mv_) == mv

    def test_stata_variable(self):
        dta = Dta117(_get_listified_domain(self.domain))

        actual_values = stata_math.st_lngamma(dta.var0_)
        expected_values = list(map(_get_math_function('lngamma'), self.domain))

        for av, ev in zip(actual_values, expected_values):
            assert av == ev, \
                "st_lngamma fails to calculate stata variable values"

    def test_normal_values(self):
        actual_values = map(stata_math.st_lngamma, self.domain)
        expected_values = list(map(_get_math_function('lngamma'), self.domain))

        for av, ev in zip(actual_values, expected_values):
            assert av == ev, \
                "st_lngamma fails to calculate normal values"


class Test_log10(object):
    domain = linspace(1, 100, 3000)

    @classmethod
    def setup_class(cls):
        raise SkipTest

    def test_missing_values(self):
        for mv_ in mvs:
            assert stata_math.st_log10(mv_) == mv

    def test_stata_variable(self):
        dta = Dta117(_get_listified_domain(self.domain))

        actual_values = stata_math.st_log10(dta.var0_)
        expected_values = list(map(_get_math_function('log10'), self.domain))

        for av, ev in zip(actual_values, expected_values):
            assert av == ev, \
                "st_log10 fails to calculate stata variable values"

    def test_normal_values(self):
        actual_values = map(stata_math.st_log10, self.domain)
        expected_values = list(map(_get_math_function('log10'), self.domain))

        for av, ev in zip(actual_values, expected_values):
            assert av == ev, \
                "st_log10 fails to calculate normal values"


class Test_sign(object):
    domain = linspace(-100, 100, 3000)

    def test_missing_values(self):
        for mv_ in mvs:
            assert stata_math.st_sign(mv_) == mv

    def test_stata_variable(self):
        dta = Dta117(_get_listified_domain(self.domain))

        actual_values = stata_math.st_sign(dta.var0_)
        expected_values = list(map(_get_math_function('sign'), self.domain))

        for av, ev in zip(actual_values, expected_values):
            assert av == ev, \
                "st_sign fails to calculate stata variable values"

    def test_normal_values(self):
        actual_values = map(stata_math.st_sign, self.domain)
        expected_values = list(map(_get_math_function('sign'), self.domain))

        for av, ev in zip(actual_values, expected_values):
            assert av == ev, \
                "st_sign fails to calculate normal values"


class Test_sin(object):
    domain = linspace(-100, 100, 3000)

    def test_missing_values(self):
        for mv_ in mvs:
            assert stata_math.st_sin(mv_) == mv

    def test_stata_variable(self):
        dta = Dta117(_get_listified_domain(self.domain))

        actual_values = stata_math.st_sin(dta.var0_)
        expected_values = list(map(_get_math_function('sin'), self.domain))

        for av, ev in zip(actual_values, expected_values):
            assert av == ev, \
                "st_sin fails to calculate stata variable values"

    def test_normal_values(self):
        actual_values = map(stata_math.st_sin, self.domain)
        expected_values = list(map(_get_math_function('sin'), self.domain))

        for av, ev in zip(actual_values, expected_values):
            assert av == ev, \
                "st_sin fails to calculate normal values"


class Test_sinh(object):
    domain = linspace(-100, 100, 3000)

    def test_missing_values(self):
        for mv_ in mvs:
            assert stata_math.st_sinh(mv_) == mv

    def test_stata_variable(self):
        dta = Dta117(_get_listified_domain(self.domain))

        actual_values = stata_math.st_sinh(dta.var0_)
        expected_values = list(map(_get_math_function('sinh'), self.domain))

        for av, ev in zip(actual_values, expected_values):
            assert av == ev, \
                "st_sinh fails to calculate stata variable values"

    def test_normal_values(self):
        actual_values = map(stata_math.st_sinh, self.domain)
        expected_values = list(map(_get_math_function('sinh'), self.domain))

        for av, ev in zip(actual_values, expected_values):
            assert av == ev, \
                "st_sinh fails to calculate normal values"


class Test_sqrt(object):
    domain = linspace(0, 100, 3000)

    def test_missing_values(self):
        for mv_ in mvs:
            assert stata_math.st_sqrt(mv_) == mv

    def test_stata_variable(self):
        dta = Dta117(_get_listified_domain(self.domain))

        actual_values = stata_math.st_sqrt(dta.var0_)
        expected_values = list(map(_get_math_function('sqrt'), self.domain))

        for av, ev in zip(actual_values, expected_values):
            assert av == ev, \
                "st_sqrt fails to calculate stata variable values"

    def test_normal_values(self):
        actual_values = map(stata_math.st_sqrt, self.domain)
        expected_values = list(map(_get_math_function('sqrt'), self.domain))

        for av, ev in zip(actual_values, expected_values):
            assert av == ev, \
                "st_sqrt fails to calculate normal values"


class Test_tan(object):
    domain = linspace(-100, 100, 3000)

    def test_missing_values(self):
        for mv_ in mvs:
            assert stata_math.st_tan(mv_) == mv

    def test_stata_variable(self):
        dta = Dta117(_get_listified_domain(self.domain))

        actual_values = stata_math.st_tan(dta.var0_)
        expected_values = list(map(_get_math_function('tan'), self.domain))

        for av, ev in zip(actual_values, expected_values):
            assert av == ev, \
                "st_tan fails to calculate stata variable values"

    def test_normal_values(self):
        actual_values = map(stata_math.st_tan, self.domain)
        expected_values = list(map(_get_math_function('tan'), self.domain))

        for av, ev in zip(actual_values, expected_values):
            assert av == ev, \
                "st_tan fails to calculate normal values"


class Test_tanh(object):
    domain = linspace(-100, 100, 3000)

    def test_missing_values(self):
        for mv_ in mvs:
            assert stata_math.st_tanh(mv_) == mv

    def test_stata_variable(self):
        dta = Dta117(_get_listified_domain(self.domain))

        actual_values = stata_math.st_tanh(dta.var0_)
        expected_values = list(map(_get_math_function('tanh'), self.domain))

        for av, ev in zip(actual_values, expected_values):
            assert av == ev, \
                "st_tanh fails to calculate stata variable values"

    def test_normal_values(self):
        actual_values = map(stata_math.st_tanh, self.domain)
        expected_values = list(map(_get_math_function('tanh'), self.domain))

        for av, ev in zip(actual_values, expected_values):
            assert av == ev, \
                "st_tanh fails to calculate normal values"



class TestAtan2(object):

    def test_two_stata(self):
        domain1 = linspace(-100, 100, 3000)
        domain2 = linspace(100, -100, 3000)
        x = Dta117(_get_listified_domain(domain1))
        y = Dta117(_get_listified_domain(domain2))

        av = stata_math.st_atan2(x.var0_, y.var0_)[:3000]
        ev = list(map(math.atan2, domain1, domain2))

        for a, e in zip(av, ev):
            if not (a == mv or round(a-e, NUM_PLACES) == 0):
                assert False, "st_math.atan2 fails to handle variables"
        assert True

    def test_stata_mv(self):
        domain = linspace(-100, 100, 3000)
        x = Dta117(_get_listified_domain(domain))

        assert stata_math.st_atan2(x.var0_, mv)[:3000] == [mv]*3000

    def test_stata_int(self):
        domain = linspace(-100, 100, 3000)
        x = Dta117(_get_listified_domain(domain))

        ev = list(map(lambda x: math.atan2(x, 3), domain))

        assert stata_math.st_atan2(x.var0_, 3)[:3000] == ev

    def test_mv_stata(self):
        domain = linspace(-100, 100, 3000)
        x = Dta117(_get_listified_domain(domain))

        assert stata_math.st_atan2(mv, x.var0_)[:3000] == [mv]*3000

    def test_int_stata(self):
        domain = linspace(-100, 100, 3000)
        x = Dta117(_get_listified_domain(domain))

        ev = list(map(lambda x: math.atan2(3, x), domain))

        assert stata_math.st_atan2(3, x.var0_)[:3000] == ev

    def test_int_int(self):
        assert stata_math.st_atan2(3, 4) == math.atan2(3, 4)


# class TestAtanH(object):

#     def test_
# def st_atanh(x):
#     """Inverse hyperbolic tangent function.
    
#     Parameters
#     ----------
#     x : float, int, MissingValue instance, or None
    
#     Returns
#     -------
#     Inverse hyperbolic tangent when x is non-missing, 
#     MISSING (".") otherwise.
    
#     """
#     if isinstance(x, StataVarVals):
#         return StataVarVals([
#             mv if _is_missing(v) or not -1 < v < 1
#             else math.atanh(v) for v in x.values
#         ])
#     if _is_missing(x) or not -1 < x < 1:
#         return mv
#     return math.atanh(x)
# #ceil
# #cloglog
# #comb
# #digamma
# #floor
# #int
# #lnfactorial
# #logit
# #max
# #min
# #mod
# #reldiff
# #round
# #sum
# #trigamma
