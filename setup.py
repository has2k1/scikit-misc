"""
Onelib

A hodge-podge collection of scientific algorithms
missing from scipy.
"""
import os
import sys
import subprocess
from setuptools import find_packages
from numpy.distutils.core import setup

import versioneer

__author__ = 'Hassan Kibirige'
__email__ = 'has2k1@gmail.com'
__description__ = "A collection of scientific algorithms."
__license__ = 'BSD (3-clause)'
__url__ = 'https://github.com/has2k1/onelib'


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
                         'onelib'],
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

    config.add_subpackage('onelib')
    config.version = versioneer.get_version()
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
    required = []
    numpy_requirement = 'numpy>=1.6.2'

    try:
        import numpy
    except Exception:
        required.append(numpy_requirement)
    else:
        if parse_version(numpy.__version__) < parse_version('1.6.2'):
            required.append(numpy_requirement)

    return required


if __name__ == '__main__':
    check_dependencies()
    prepare_for_setup()

    setup(name='onelib',
          maintainer=__author__,
          maintainer_email=__email__,
          description=__description__,
          long_description=__doc__,
          license=__license__,
          version=versioneer.get_version(),
          cmdclass=versioneer.get_cmdclass(),
          url=__url__,
          install_requires=get_required_packages(),
          setup_requires=setup_requires(),
          packages=find_packages(),
          package_data=get_package_data(),
          classifiers=[
              'Intended Audience :: Science/Research',
              'License :: OSI Approved :: BSD License',
              'Programming Language :: Python :: 2',
              'Programming Language :: Python :: 3',
              'Topic :: Scientific/Engineering',
          ],
          configuration=configuration
          )
