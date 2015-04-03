import sys

_str = str
if sys.version.startswith('3'):
	str = _str
else:
	str = unicode

def is_string(var):
	return isinstance(var, (str, _str))
