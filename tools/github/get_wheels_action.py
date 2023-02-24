#!/usr/bin/env python

# Determines what to do when the build-wheels action is running
# Options are:
#     - build
#     - build, release
#     - nothing
# One of these values is printed to the standard output.
import re

from _repo import Repo, GithubInfo

# Define a releasable version to be valid according to PEP440
# and is a semver
count = r"(?:[0-9]|[1-9][0-9]+)"

VERSION_TAG_PATTERN = re.compile(
    r"^v"
    rf"{count}\.{count}\.{count}"
    # optional pre-release part
    r"(?:"
    rf"(?:a|b|rc|alpha|beta){count}"
    r")?"
    r"$"
)

VERSION_TAG_MESSAGE_PATTERN = re.compile(
    r"^Version: " +
    VERSION_TAG_PATTERN.pattern[2:]
)


BUILD_PATTERN = re.compile(
    r"\[wheel build\]$"
)


def is_wheel_build(repo: Repo) -> bool:
    """
    Return True commit title ends in "[wheel build]"
    """
    title = repo.commit_titles(1)[0]
    return bool(BUILD_PATTERN.search(title))


def is_version_tag(repo: Repo) -> bool:
    """
    Return True if the ref is a tag that is a releasible version
    """
    if repo.info.ref_type != "tag":
        return False
    return bool(VERSION_TAG_PATTERN.match(repo.info.ref_name))


def is_version_tag_message(repo: Repo) -> bool:
    """
    Return True if tag message is of the form "Version: 0.1.0"
    """
    if repo.info.ref_type != "tag":
        return False
    tag_msg = repo.tag_message(repo.info.ref_name)
    return bool(VERSION_TAG_MESSAGE_PATTERN.match(tag_msg))


if __name__ == "__main__":
    repo = Repo(GithubInfo())
    build_cmd = is_wheel_build(repo)
    version_tag = is_version_tag(repo) or is_version_tag_message(repo)

    if version_tag:
        print("build, release")
    elif build_cmd:
        print("build")
    else:
        print("nothing")
