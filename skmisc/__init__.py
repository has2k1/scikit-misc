from ._version import get_versions

__version__ = get_versions()['version']
del get_versions

__all__ = ['__version__']

# We first need to detect if we're being called as part of the skmisc
# setup procedure itself in a reliable manner.
try:
    __SKMISC_SETUP__
except NameError:
    __SKMISC_SETUP__ = False

if __SKMISC_SETUP__:
    import sys as _sys
    _sys.stderr.write('Running from skmisc source directory.\n')
    del _sys
else:
    try:
        from skmisc.__config__ import show as show_config  # noqa: F401
    except ImportError:
        msg = """Error importing skmisc: you cannot import skmisc while
        being in skmisc source directory; please exit the skmisc source
        tree first, and relaunch your python intepreter."""
        raise ImportError(msg)

    __all__.append('show_config')

    def test(args=None, plugins=None):
        """
        Run tests
        """
        import pytest
        return pytest.main(args=args, plugins=plugins)
