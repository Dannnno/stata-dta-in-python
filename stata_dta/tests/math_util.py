import math

from ..stata_missing import MISSING as mv

def cloglog(x):
    log = math.log
    return log(-log(1 - x))


def invcloglog(x):
    exp = math.exp
    return 1 - exp(-exp(x))


def invlogit(x):
    return math.exp(x)/(1 + math.exp(x))


def ln(x):
    return math.log(x)


def lngamma(x):
    return min(mv, math.lgamma(x))


def log10(x):
    return math.log(x, 10)


def sign(x):
    return 0 if x == 0 else -1 if x < 0 else 1
