#!/usr/bin/env python

# Determines what to do when the build-wheels action is running
# Options are:
#     - :build:
#     - :build:release:
#     - :build:pre_release:
#     - :nothing:
# One of these values is printed to the standard output.
# Testing:
#   GITHUB_REF_NAME="v0.2.0a1" GITHUB_REF_TYPE="tag" ./get_wheels_action.py
#   GITHUB_REF_NAME="v0.2.0" GITHUB_REF_TYPE="tag" ./get_wheels_action.py

import os
import re
import sys

from _repo import Git, Workspace

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


if __name__ == "__main__":
    ws = Workspace()
    tag = ws.head_tag()

    build = head_commit_wants_wheel_build()
    release = (
        Git.is_annotated(tag) and
        is_release_tag(tag) and
        is_release_tag_message_ok(tag)
    )
    pre_release = (
        Git.is_annotated(tag) and
        is_pre_release_tag(tag) and
        is_release_tag_message_ok(tag)
    )

    if release:
        print(":build:release:")
    elif pre_release:
        print(":build:pre_release:")
    elif build:
        print(":build:")
    else:
        print(":nothing:")
