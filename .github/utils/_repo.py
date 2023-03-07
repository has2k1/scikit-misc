#!/usr/bin/env python
from __future__ import annotations

import os
import re
import shlex
from subprocess import Popen, PIPE

from typing import Sequence

# https://docs.github.com/en/actions/learn-github-actions/variables
# #default-environment-variables
GITHUB_VARS = [
    "GITHUB_REF_NAME",  # main, dev, v0.1.0, v0.1.3a1
    "GITHUB_REF_TYPE",  # "branch" or "tag"
    "GITHUB_REPOSITORY",  # has2k1/scikit-misc
    "GITHUB_SERVER_URL",  # https://github.com
    "GITHUB_SHA",  # commit shasum
    "GITHUB_WORKSPACE",  # /home/runner/work/scikit-misc/scikit-misc
    "GITHUB_EVENT_NAME"  # push, schedule, workflow_dispatch, ...
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


class Git:
    @staticmethod
    def checkout(committish):
        """
        Return True if inside a git repo
        """
        res = run(f"git checkout {committish}")
        return res

    @staticmethod
    def commit_titles(n=1) -> list[str]:
        """
        Return a list n of commit titles
        """
        output = run(
            f"git log --oneline --no-merges --pretty='format:%s' -{n}"
        )
        return output.split("\n")

    @staticmethod
    def head_commit_title() -> str:
        """
        Commit title
        """
        return Git.commit_titles(1)[0]

    @staticmethod
    def is_repo():
        """
        Return True if inside a git repo
        """
        res = run("git rev-parse --is-inside-work-tree")
        return res == "return"

    @staticmethod
    def fetch_tags() -> str:
        """
        Fetch all tags
        """
        return run("git fetch --tags --force")

    @staticmethod
    def is_shallow() -> bool:
        """
        Return True if current repo is shallow
        """
        res = run("git rev-parse --is-shallow-repository")
        return res == "true"

    @staticmethod
    def deepen(n: int = 1) -> str:
        """
        Fetch n commits beyond the shallow limit
        """
        return run(f"git fetch --deepen={n}")

    @staticmethod
    def describe() -> str:
        """
        Git describe to determine version
        """
        return run("git describe --dirty --tags --long --match '*[0-9]*'")

    @staticmethod
    def can_describe() -> bool:
        """
        Return True if repo can be "described" from a semver tag
        """
        return bool(DESCRIBE_PATTERN.match(Git.describe()))

    @staticmethod
    def tag_message(tag: str) -> str:
        """
        Get the message of a tag
        """
        return run(f"git tag -l --format='%(subject)' {tag}")

    @staticmethod
    def is_annotated(tag: str) -> bool:
        """
        Return true if tag is annotated tag
        """
        # LHS prints to stderr and returns nothing when
        # tag is an empty string
        return run(f"git cat-file -t {tag}") == "tag"

    @staticmethod
    def shallow_checkout(branch: str, url: str, depth: int = 1) -> str:
        """
        Shallow clone upto n commits
        """
        _branch = f"--branch={branch}"
        _depth = f"--depth={depth}"
        return run(f"git clone {_depth} {_branch} {url} .")

    @staticmethod
    def head_tag() -> str:
        """
        Return tag at HEAD or empty string if there is none
        """
        tags = run("git tag --points-at HEAD").split("\n")
        return tags[0]


class Workspace:
    """
    Github Actions workspace information about the repository and action
    """
    # From github environment
    event_name: str
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
            setattr(self, param, os.environ.get(name))

        self.repo_url = f"{self.server_url}/{self.repository}.git"

    def ref_is_tag(self) -> bool:
        """
        Return true if ref (HEAD) is a tag
        """
        return self.ref_type == "tag"

    def is_push_event(self) -> bool:
        """
        Return True if push triggered the action
        """
        return self.event_name == "push"

    def pushed_tag(self) -> str:
        """
        Return pushed tag or empty string it isn't a tag
        """
        return self.ref_name if self.ref_is_tag() else ""
