#!/usr/bin/env python

import os
import re
import shlex
from subprocess import Popen, PIPE

from typing import Sequence

DEFAULT_DESCRIBE = [
    "git",
    "describe",
    "--dirty",
    "--tags",
    "--long",
    "--match",
    "*[0-9]*",
]

LAST_COMMIT_SHORTHASH = [
    "git",
    "log",
    "--format='%h'",
    "-n",
    "1",
]

DESCRIBE_PATTERN = re.compile(
    r"^"
    r"v"
    r"(?P<version>\d+\.\d+\.\d+)"
    r"(-(?P<commits>\d+)-g(?P<hash>[a-z0-9]+))?"
    r"(?P<dirty>-dirty)?"
    r"$"
)


def run(cmd: str | Sequence[str]) -> str:
    if isinstance(cmd, str) and os.name == "posix":
        cmd = shlex.split(cmd)
    with Popen(
        cmd,
        stdin=PIPE,
        stderr=PIPE,
        stdout=PIPE,
        text=True,
        encoding="utf-8"
    ) as p:
        stdout, _ = p.communicate()
    return stdout.strip()


def get_version() -> str:
    """
    Return a SemVer (& PEP440 compliant) from `git describe` output
    """
    desc = run(DEFAULT_DESCRIBE)
    m = DESCRIBE_PATTERN.match(desc)

    if not m:
        return "0.0.0"

    info = m.groupdict()
    if "commits" in info:
        tpl = "{version}.dev{commits}"
    else:
        tpl = "{version}"

    v = tpl.format(**info)
    return v


def get_last_commit_shorthash() -> str:
    h = run(LAST_COMMIT_SHORTHASH)
    return h


if __name__ == "__main__":
    print(get_version())
