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
    except ImportError as err:
        msg = """Error importing skmisc: you cannot import skmisc while
        being in skmisc source directory; please exit the skmisc source
        tree first, and relaunch your python intepreter."""
        raise ImportError('\n\n'.join([err.message, msg]))

    __all__.append('show_config')

    def test(args=None, plugins=None):
        """
        Run tests
        """
        # The doctests are not run when called from an installed
        # package since the pytest.ini is not included in the
        # package.
        import os
        try:
            import pytest
        except ImportError:
            msg = "To run the tests, you must install pytest"
            raise ImportError(msg)
        path = os.path.dirname(os.path.realpath(__file__))
        if args is None:
            args = [path]
        else:
            args.append(path)
        return pytest.main(args=args, plugins=plugins)
