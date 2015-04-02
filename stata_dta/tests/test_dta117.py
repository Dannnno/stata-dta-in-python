from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

from contextlib import contextmanager

import io
import csv
import os
import sys

from nose.plugins.skip import SkipTest

from .. import open_dta, Dta115, Dta117


examples_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'examples')


@contextmanager
def capture():
    try:
        out = [io.StringIO(), io.StringIO()]
        oldout, olderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = out

        yield out

    finally:
        out[0], out[1] = out[0].getvalue(), out[1].getvalue()
        sys.stdout, sys.stderr = oldout, olderr


def test_iterable_to_dta117_list():
    data = [[0.0, 0.1, 0.2], [1.0, 1.1, 1.2], [2.0, 2.1, 2.2]]
    dta = Dta117(data)

    expected_list = """
    +----------------------------------+
    |     var0        var1        var2 |
    +----------------------------------+
 0. |        0         0.1         0.2 |
 1. |        1         1.1         1.2 |
 2. |        2         2.1         2.2 |
    +----------------------------------+
""".lstrip('\n')

    with capture() as out:
        dta.list()

    assert expected_list == out[0], out[0]
