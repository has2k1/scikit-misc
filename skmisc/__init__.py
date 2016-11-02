from ._version import get_versions

try:
    from skmisc.__config__ import show as show_config
except ImportError:
    msg = """Error importing skmisc: you cannot import skmisc while
    being in skmisc source directory; please exit the skmisc source
    tree first, and relaunch your python intepreter."""
    raise ImportError(msg)

__version__ = get_versions()['version']
del get_versions


def test(args=None, plugins=None):
    """
    Run tests
    """
    import pytest
    return pytest.main(args=args, plugins=plugins)
