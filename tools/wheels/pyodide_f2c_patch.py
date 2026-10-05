#!/usr/bin/env python
"""
Adapt the loess C sources to the f2c translation of the Fortran sources

Pyodide translates Fortran to C with f2c. Compared with gfortran, f2c
makes every subroutine return `int` and passes a hidden length argument
after the arguments of a routine that has a `CHARACTER` argument.
WebAssembly checks function signatures at link time, so the C
declarations and definitions of the Fortran routines must agree with
that translation.

The script edits the sources in place and is safe to run repeatedly. It
raises if a source no longer has the text it rewrites.

Usage: pyodide_f2c_patch.py [PROJECT_DIR]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

LOESS_SRC = Path("skmisc/loess/src")

# The libf2c routines that the translated Fortran calls, next to this script
RUNTIME_SOURCE = "f2c_runtime.c"

# C routines that the Fortran calls back into
CALLBACKS = ("ehg182", "ehg183a", "ehg184a")

# Callbacks that take a `CHARACTER` argument, with the last argument
# before the hidden length that f2c adds
CHARACTER_CALLBACKS = {"ehg183a": "inc", "ehg184a": "inc"}


def rewrite(text: str, pattern: str, repl: str, expected: int) -> str:
    """
    Replace every match of a pattern, requiring an exact number of matches

    Parameters
    ----------
    text
        Source text to rewrite.
    pattern
        Regular expression, matched per line.
    repl
        Replacement text.
    expected
        Number of matches the source must contain.

    Returns
    -------
    str
        The rewritten text.

    Raises
    ------
    RuntimeError
        If the number of matches differs from `expected`.
    """
    result, n = re.subn(pattern, repl, text, flags=re.MULTILINE)
    if n != expected:
        raise RuntimeError(
            f"Expected {expected} matches of {pattern!r}, found {n}"
        )
    return result


def return_int_from_declarations(text: str, expected: int) -> str:
    """
    Declare the Fortran routines as returning `int`

    Parameters
    ----------
    text
        Source text of a C file.
    expected
        Number of declarations and definitions starting with `void`.

    Returns
    -------
    str
        The rewritten text.
    """
    return rewrite(text, r"^void( F77_SUB)", r"int\1", expected)


def patch_loess_c(text: str) -> str:
    """
    Rewrite `loess.c`

    Parameters
    ----------
    text
        Source text of `loess.c`.

    Returns
    -------
    str
        The rewritten text.
    """
    return return_int_from_declarations(text, expected=2)


def patch_loessc_c(text: str) -> str:
    """
    Rewrite `loessc.c`

    Parameters
    ----------
    text
        Source text of `loessc.c`.

    Returns
    -------
    str
        The rewritten text.
    """
    # Prototypes of 12 routines, and the definitions of 2 callbacks
    text = return_int_from_declarations(text, expected=14)
    # The definition of ehg182 has its return type on a line of its own
    text = rewrite(text, r"^void(\nF77_SUB\(ehg182\))", r"int\1", 1)

    # Callbacks are called from Fortran and the value is never used, but
    # a function that returns `int` must still return.
    for name in CALLBACKS:
        text = rewrite(
            text,
            rf"(F77_SUB\({name}\)\([^;{{]*\)\n\{{(?:.|\n)*?)\n\}}\n",
            r"\1\n    return 0;\n}\n",
            1,
        )

    for name, last in CHARACTER_CALLBACKS.items():
        text = rewrite(
            text,
            rf"(F77_SUB\({name}\)\([^)]*\*{last})\)",
            r"\1, int s_len)",
            2,
        )
    return text


def is_patched(text: str) -> bool:
    """
    Report whether `loessc.c` has already been rewritten

    Parameters
    ----------
    text
        Source text of `loessc.c`.

    Returns
    -------
    bool
        `True` if no Fortran routine is declared as returning `void`.
    """
    return re.search(r"^void F77_SUB", text, flags=re.MULTILINE) is None


def main(project_dir: Path) -> None:
    """
    Patch the loess sources under a project directory

    Parameters
    ----------
    project_dir
        Root of the scikit-misc source tree.
    """
    loessc = project_dir / LOESS_SRC / "loessc.c"
    loess = project_dir / LOESS_SRC / "loess.c"
    runtime = Path(__file__).with_name(RUNTIME_SOURCE)

    if is_patched(loessc.read_text()):
        print("loess sources already patched for f2c")
        return

    # Pyodide has no libf2c, so compile the routines the translation calls
    # into the extension module that contains the Fortran callers.
    loessc.write_text(
        patch_loessc_c(loessc.read_text()) + "\n" + runtime.read_text()
    )
    loess.write_text(patch_loess_c(loess.read_text()))
    print("patched loess sources for f2c")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "."))
