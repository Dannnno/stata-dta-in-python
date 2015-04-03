from contextlib import contextmanager
import io
import sys

from nose.plugins.skip import SkipTest


def raises(lambda_, exception=Exception, *args, **kwargs):
	try:
		lambda_(*args, **kwargs)
	except exception:
		return True
	else:
		return False


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


def skip_if_version(version):
	if sys.version.startswith(version):
		raise SkipTest