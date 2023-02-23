#!/usr/bin/env python

import re

from _repo import Repo, GithubInfo

# Define a releasable version to be valid according to PEP440
# and is a semver
count = r"(?:[0-9]|[1-9][0-9]+)"
VERSION_PATTERN = re.compile(
    r"^v"
    rf"{count}\.{count}\.{count}"
    # optional pre-release part
    r"(?:"
    rf"(?:a|b|rc|alpha|beta){count}"
    r")?"
    r"$"
)


def is_good_version_tag() -> bool:
    """
    Return True if the ref is a tag that is a releasible version
    """
    repo = Repo(GithubInfo())
    if repo.info.ref_type != "tag":
        return False
    return bool(VERSION_PATTERN.match(repo.info.ref_name))


if __name__ == "__main__":
    if is_good_version_tag():
        print("Yes")
    else:
        print("No")
