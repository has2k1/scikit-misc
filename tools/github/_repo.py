#!/usr/bin/env python

import os
import re
import shlex
from subprocess import Popen, PIPE

from typing import Sequence

# https://docs.github.com/en/actions/learn-github-actions/variables
# #default-environment-variables
GITHUB_VARS = [
    "GITHUB_REF_NAME",  # branch-cool or v0.1.0
    "GITHUB_REF_TYPE",  # "branch" or "tag"
    "GITHUB_REPOSITORY",  # user/repo
    "GITHUB_SERVER_URL",  # https://github.com
    "GITHUB_SHA",  # commit shasum
    "GITHUB_WORKSPACE"  # /home/runner/work/repo-name/repo-name
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


class GithubInfo:
    """
    Github information about the repository and action
    """
    # From github environment
    ref_name: str
    ref_type: str
    repository: str
    server_url: str
    sha: str
    workspace: str

    # Derived
    repo_url: str

    def __init__(self):
        for name in GITHUB_VARS:
            param = name.replace("GITHUB_", "").lower()
            setattr(self, param, os.getenv(name))

        self.repo_url = f"{self.server_url}/{self.repository}.git"


class Repo:
    info: GithubInfo

    def __init__(self, info: GithubInfo):
        self.info = info

    def is_git(self):
        """
        Return True if inside a git repo
        """
        res = run("git rev-parse --is-inside-work-tree")
        return res == "return"

    def shallow_checkout(self, n: int = 1) -> str:
        """
        Shallow clone upto n commits
        """
        branch = f"--branch={self.info.ref_name}"
        depth = f"--depth={n}"
        return run(f"git clone {depth} {branch} {self.info.repo_url} .")

    def commit_titles(self, n=1) -> list[str]:
        output = run(
            f"git log --oneline --no-merges --pretty='format:%s' -{n}"
        )
        return output.split("\n")


    def is_shallow(self) -> bool:
        """
        Return True if current repo is shallow
        """
        res = run("git rev-parse --is-shallow-repository")
        return res == "true"

    def deepen(self, n: int = 1) -> str:
        """
        Fetch n commits beyond the shallow limit
        """
        return run(f"git fetch --deepen={n}")

    def describe(self) -> str:
        """
        Git describe to determine version
        """
        return run("git describe --dirty --tags --long --match '*[0-9]*'")

    def can_describe(self) -> bool:
        """
        Return True if repo can be "described" from a semver tag
        """
        return bool(DESCRIBE_PATTERN.match(self.describe()))

    def tag_message(self, tag: str) -> str:
        """
        Get the message of an annotated tag
        """
        return run(f"git tag -l --format='%(contents)' {tag}")
