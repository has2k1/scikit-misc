"""
scikit-misc

Miscellaneous tools for data analysis and scientific computing.
"""
import os
import sys
import subprocess

import versioneer

if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins

__author__ = 'Hassan Kibirige'
__email__ = 'has2k1@gmail.com'
__description__ = "Miscellaneous tools for scientific computing."
__license__ = 'BSD (3-clause)'
__url__ = 'https://github.com/has2k1/scikit-misc'

# BEFORE importing setuptools, remove MANIFEST. Otherwise it may
# not be properly updated when the contents of directories change
# (true for distutils, not sure about setuptools).
if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')

# This is a bit hackish: we are setting a global variable so that
# the main skmisc __init__ can detect if it is being loaded by the
# setup routine, to avoid attempting to load components that aren't
# built yet. Copied from numpy
builtins.__SKMISC_SETUP__ = True


def check_dependencies():
    """
    Check for system level dependencies
    """
    pass


def get_required_packages():
    """
    Return required packages

    Plus any version tests and warnings
    """
    install_requires = ['numpy']
    return install_requires


def get_package_data():
    """
    Return package data

    For example:

        {'': ['*.txt', '*.rst'],
         'hello': ['*.msg']}

    means:
        - If any package contains *.txt or *.rst files,
          include them
        - And include any *.msg files found in
          the 'hello' package, too:
    """
    return {}


def generate_cython():
    cwd = os.path.abspath(os.path.dirname(__file__))
    print("Cythonizing sources")
    p = subprocess.call([sys.executable,
                         os.path.join(cwd, 'tools', 'cythonize.py'),
                         'skmisc'],
                        cwd=cwd)
    if p != 0:
        raise RuntimeError("Running cythonize failed!")


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('skmisc')
    return config


def prepare_for_setup():
    cwd = os.path.abspath(os.path.dirname(__file__))
    if not os.path.exists(os.path.join(cwd, 'PKG-INFO')):
        # Generate Cython sources, unless building from source release
        generate_cython()


def setup_requires():
    """
    Return required packages

    Plus any version tests and warnings
    """
    from pkg_resources import parse_version
    required = ['cython>=0.24.0']
    numpy_requirement = 'numpy>=1.7.1'

    try:
        import numpy
    except Exception:
        required.append(numpy_requirement)
    else:
        if parse_version(numpy.__version__) < parse_version('1.7.1'):
            required.append(numpy_requirement)

    return required


def setup_package():
    from setuptools import find_packages
    # versioneer needs numpy cmdclass
    from numpy.distutils.core import setup, numpy_cmdclass
    metadata = dict(
        name='scikit-misc',
        maintainer=__author__,
        maintainer_email=__email__,
        description=__description__,
        long_description=__doc__,
        license=__license__,
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(numpy_cmdclass),
        url=__url__,
        install_requires=get_required_packages(),
        setup_requires=setup_requires(),
        packages=find_packages(),
        package_data=get_package_data(),
        classifiers=[
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: BSD License',
            'Operating System :: Unix',
            'Operating System :: MacOS',
            'Programming Language :: C',
            'Programming Language :: Fortran',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 3',
            'Topic :: Scientific/Engineering',
        ],
        configuration=configuration
    )
    setup(**metadata)


if __name__ == '__main__':
    check_dependencies()
    prepare_for_setup()
    setup_package()

    del builtins.__SKMISC_SETUP__
