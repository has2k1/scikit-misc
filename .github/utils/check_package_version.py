#!/usr/bin/env python
# Exit 1 if wheel has a bad ver
import sys
from pathlib import Path

PACKAGE_NAME = "scikit_misc"
NULL_VERSION = "0.0.0"
BAD_PFX = f"{PACKAGE_NAME}-{NULL_VERSION}"

WHEELS = Path("wheelhouse/").glob("*.whl")
DISTS = Path("dist/").glob("*.tar.gz")


def wheels_have_good_version() -> bool:
    """
    Return True if there is a wheel with a null version
    """
    return all(not s.name.startswith(BAD_PFX) for s in WHEELS)


def sdists_have_good_version() -> bool:
    """
    Return True if there is an sdist with a null version
    """
    return all(not s.name.startswith(BAD_PFX) for s in DISTS)


def have_good_versions() -> bool:
    """
    Return True if any wheel or dist has a null version

    A null version is "0.0.0"
    """
    return wheels_have_good_version() and sdists_have_good_version()


if __name__ == "__main__":
    if not have_good_versions():
        sys.exit(1)
