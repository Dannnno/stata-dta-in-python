from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import csv
import os

from nose.plugins.skip import SkipTest

from .. import open_dta, Dta115
from .util import capture, raises, skip_if_version


examples_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'examples')


def test_iterable_to_dta115_list():
    data = [[0.0, 0.1, 0.2], [1.0, 1.1, 1.2], [2.0, 2.1, 2.2]]
    dta = Dta115(data)

    expected = """
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

    assert expected == out[0], out[0]


def test_save_to_file_no_name():
    data = [[0.0, 0.1, 0.2], [1.0, 1.1, 1.2], [2.0, 2.1, 2.2]]
    dta = Dta115(data)
    assert raises(dta.save, ValueError)


def test_save_to_file_with_name():
    skip_if_version('3')
    data = [[0.0, 0.1, 0.2], [1.0, 1.1, 1.2], [2.0, 2.1, 2.2]]
    dta = Dta115(data)
    try:
        dta.save('tempfile.dta')
    except (IOError,    # file exists and replace not selected Python2
            OSError     # file exists and replace not selected Python3
            ) as e:
        assert False
    else:
        assert os.path.exists('tempfile.dta')
    finally:
        os.remove('tempfile.dta')


def test_save_to_file_name_exists():
    skip_if_version('3')
    data = [[0.0, 0.1, 0.2], [1.0, 1.1, 1.2], [2.0, 2.1, 2.2]]
    dta = Dta115(data)
    try:
        dta.save('tempfile2.dta')
    except (IOError,    # file exists and replace not selected Python2
            OSError     # file exists and replace not selected Python3
            ) as e:
        assert False
    else:
        assert not raises(
            dta.save, (IOError, OSError), replace=True)
    finally:
        os.remove('tempfile2.dta')


def test_access_data():
    raise SkipTest
    data = [[0.0, 0.1, 0.2], [1.0, 1.1, 1.2], [2.0, 2.1, 2.2]]
    dta = Dta115(data)

    print(dta.var0_)
    assert False, dta.var0_
