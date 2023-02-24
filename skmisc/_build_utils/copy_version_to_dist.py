#!/usr/bin/env python
import os
import shutil
from pathlib import Path

dist_root: str = os.getenv("MESON_DIST_ROOT")  # type: ignore
build_root: str = os.getenv("MESON_BUILD_ROOT")  # type: ignore


def copy_to_dist(rel_path: str) -> None:
    """
    Copy file at rel_path to the sdist directory
    """
    src = Path(build_root) / rel_path
    dest = Path(dist_root) / rel_path
    if src.exists():
        shutil.copy(src, dest)


if __name__ == "__main__":
    copy_to_dist("skmisc/_version.py")
