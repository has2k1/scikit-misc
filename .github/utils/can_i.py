#!/usr/bin/env python

# Determines what to do when the build-wheels action is running
# If called with an argument
# Determines whether to build and/or (pre)release wheels
#   can_i.py build|release|pre_release
#
# Prints "true" or "false"
#
# One of these values is printed to the standard output.
# Testing:
#   GITHUB_REF_NAME="v0.2.0a1" GITHUB_REF_TYPE="tag" ./can_i.py build
#   GITHUB_REF_NAME="v0.2.0" GITHUB_REF_TYPE="tag" ./can_i.py release

import re
import sys

from typing import Callable, TypeAlias

from _repo import Git, Workspace


Ask: TypeAlias = Callable[[], bool]
TAG = Workspace().head_tag()

# Define a releasable version to be valid according to PEP440
# and is a semver
count = r"(?:[0-9]|[1-9][0-9]+)"

RELEASE_TAG_PATTERN = re.compile(
    r"^v"
    rf"{count}\.{count}\.{count}"
    r"$"
)

# Prerelease version
PRE_RELEASE_TAG_PATTERN = re.compile(
    r"^v"
    rf"{count}\.{count}\.{count}"
    r"(?:"
    rf"(?:a|b|rc|alpha|beta){count}"
    r")"
    r"$"
)

VERSION_TAG_MESSAGE_PREFIX = "Version "

BUILD_PATTERN = re.compile(
    r"\[wheel build\]$"
)

def head_commit_wants_wheel_build() -> bool:
    """
    Return True commit title ends in "[wheel build]"
    """
    return bool(BUILD_PATTERN.search(Git.head_commit_title()))


def is_release_tag(tag: str) -> bool:
    """
    Return True if the ref is a tag that is a releasible version
    """
    return bool(RELEASE_TAG_PATTERN.match(tag))


def is_pre_release_tag(tag: str) -> bool:
    """
    Return True if the ref is a tag that is a prereleasible version
    """
    return bool(PRE_RELEASE_TAG_PATTERN.match(tag))


def is_release_tag_message_ok(tag: str) -> bool:
    """
    Return True if tag message is of the form "Version 0.1.0"

    Check for both the release and pre-release
    """
    tag_msg = Git.tag_message(tag)
    return tag_msg == f"{VERSION_TAG_MESSAGE_PREFIX}{tag[1:]}"


def can_build() -> bool:
    """
    Return True commit message request build
    """
    return head_commit_wants_wheel_build()


def can_release() -> bool:
    """
    Return True if tag what we expect for a release
    """
    return (
        Git.is_annotated(TAG) and
        is_release_tag(TAG) and
        is_release_tag_message_ok(TAG)
    )


def can_pre_release() -> bool:
    """
    Return True if tag what we expect for a pre_release
    """
    return (
        Git.is_annotated(TAG) and
        is_pre_release_tag(TAG) and
        is_release_tag_message_ok(TAG)
    )


ACTIONS: dict[str, Ask] = {
    "build": can_build,
    "release": can_release,
    "pre_release": can_pre_release,
}

if __name__ == "__main__":
    if len(sys.argv) == 2:
        action = sys.argv[1]
        result = ACTIONS.get(action, lambda: False)()
        print(str(result).lower())  # "true", "false"
    else:
        print("false")
