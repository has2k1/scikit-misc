#!python
""" Detect bitness (32 or 64) of Mingw-w64 gcc build target on Windows.
"""
# Ref: https://github.com/scipy/scipy/blob/main/scipy/_build_utils/gcc_build_bitness.py  # noqa: E501
# commit: 6c4a3f3

import re
from subprocess import run


def main():
    res = run(
        ['gcc', '-v'],
        check=True,
        text=True,
        capture_output=True
    )
    target = re.search(
        r'^Target: (.*)$',
        res.stderr,
        flags=re.M
    ).groups()[0]

    if target.startswith('i686'):
        print('32')
    elif target.startswith('x86_64'):
        print('64')
    else:
        raise RuntimeError("Could not detect Mingw-w64 bitness")


if __name__ == "__main__":
    main()
