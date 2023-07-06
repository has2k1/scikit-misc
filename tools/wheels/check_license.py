#!/usr/bin/env python
"""
check_license.py [MODULE]

Check the presence of a LICENSE.txt in the installed module directory,
and that it appears to contain text prevalent for a scikit-misc binary
distribution.

"""
import argparse
import io
import re
import sys
from pathlib import Path


def check_text(text):
    ok = "Copyright (c)" in text and re.search(
        r"This binary distribution of \w+ also bundles the following software",
        text,
        re.IGNORECASE
    )
    return ok


def main():
    p = argparse.ArgumentParser(usage=__doc__.rstrip())
    p.add_argument("module", nargs="?", default="skmisc")
    args = p.parse_args()

    # Drop '' from sys.path
    sys.path.pop(0)

    # Find module path
    __import__(args.module)
    mod = sys.modules[args.module]

    # Check license text
    module_file = Path(mod.__file__)  # type: ignore
    license_txt = module_file.parent / "LICENSE.txt"
    with io.open(license_txt, "r", encoding="utf-8") as f:
        text = f.read()

    ok = check_text(text)
    if not ok:
        print(
            "ERROR: License text {} does not contain expected "
            "text fragments\n".format(license_txt)
        )
        print(text)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
