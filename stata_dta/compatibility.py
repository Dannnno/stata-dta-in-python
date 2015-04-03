import itertools
import sys

_str = str
if sys.version.startswith('3'):
	str = _str
	range = range
	zip = zip
	map = map
	builtin = __import__('builtins')
else:
	str = unicode
	range = xrange
	zip = itertools.izip
	map = itertools.imap
	builtin = __import__('__builtin__')


def is_string(var):
	return isinstance(var, (str, _str))


def is_bytes(var):
	return NotImplemented


def is_bytearray(var):
	return NotImplemented


def is_stringy(var):
	return is_string(var) or is_bytes(var) or is_bytearray(var)
