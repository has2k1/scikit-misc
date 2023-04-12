#!/usr/bin/env python
from __future__ import annotations

import os
import re
import shlex
from pathlib import Path
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

count = r"(?:[0-9]|[1-9][0-9]+)"
DESCRIBE_PATTERN = re.compile(
    r"^v"
    rf"(?P<version>{count}\.{count}\.{count})"
    rf"(?P<pre>(a|b|rc|alpha|beta){count})?"
    r"(-(?P<commits>\d+)-g(?P<hash>[a-z0-9]+))?"
    r"(?P<dirty>-dirty)?"
    r"$"
)

VERSION_LINE_PATTERN = re.compile(
    r'__version__ = "(?P<version>.+?)"'
)

NULL_VERSION = "0.0.0"

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


def is_git() -> bool:
    """
    Return True if inside a git repo
    """
    res = run("git rev-parse --is-inside-work-tree")
    return res == "return"


def get_version_from_scm() -> str:
    """
    Return a SemVer (& PEP440 compliant) from `git describe` output
    """
    desc = run(DEFAULT_DESCRIBE)
    m = DESCRIBE_PATTERN.match(desc)

    if not m:
        return ""

    pre = m.group("pre") or ""
    version = m.group("version")
    commits = m.group("commits")

    if commits != "0":
        v = f"{version}.dev{commits}"
    else:
        v = f"{version}{pre}"

    return v


def get_version_from_file() -> str:
    """
    Read version from package/_version.py
    """
    file = Path(__file__).parent.parent / "_version.py"

    if file.exists():
        s = file.read_text()
        match = VERSION_LINE_PATTERN.search(s)
        if match:
            return match.group('version')
    return ""


def get_version() -> str:
    """
    Return a SemVer (& PEP440 compliant) from `git describe` output
    """
    return get_version_from_scm() or get_version_from_file() or NULL_VERSION


if __name__ == "__main__":
    print(get_version())
