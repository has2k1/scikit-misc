from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version('scikit_misc')
except PackageNotFoundError:
    # package is not installed
    pass
finally:
    del version
    del PackageNotFoundError

__all__ = ['__version__']

# try:
#     from skmisc.__config__ import show as show_config  # noqa: F401
# except ImportError as err:
#     msg = """Error importing skmisc: you cannot import skmisc while
#     being in skmisc source directory; please exit the skmisc source
#     tree first, and relaunch your python intepreter."""
#     raise ImportError('\n\n'.join([err.message, msg]))
# else:
#     __all__.append('show_config')
#
#
#     def test(args=None, plugins=None):
#         """
#         Run tests
#         """
#         from pathlib import Path
#         # The doctests are not run when called from an installed
#         # package since the pytest.ini is not included in the
#         # package.
#         try:
#             import pytest
#         except ImportError:
#             msg = "To run the tests, you must install pytest"
#             raise ImportError(msg)
#         path = str(Path(__file__).parent)
#         if args is None:
#             args = [path]
#         else:
#             args.append(path)
#         return pytest.main(args=args, plugins=plugins)
