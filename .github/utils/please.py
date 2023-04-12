#!/usr/bin/env python

# This script knows what to do
#   1. please.py can_i_build|can_i_release|can_i_pre_release
#      Prints "true" or "false"
#
#   2. please.py checkout_build_commit
#
#
# One of these values is printed to the standard output.
# Testing:
#   GITHUB_REF_NAME="v0.2.0a1" GITHUB_REF_TYPE="tag" ./can_i.py build
#   GITHUB_REF_NAME="v0.2.0" GITHUB_REF_TYPE="tag" ./can_i.py release

import re
import sys
import os

from typing import Callable, TypeAlias

from _repo import Git


Ask: TypeAlias = Callable[[], bool]
Do: TypeAlias = Callable[[], str]

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

# build head | branch | commit | tag
# [wheel build]
# [wheel build: main]
# [wheel build: v0.2.1]
# [wheel build: a4689f5]
BUILD_PATTERN = re.compile(
    r"\[wheel build(: (?P<build_ref>.+?))?\]"
)


def get_build_ref() -> str:
    """
    Get reference to the commit that will be built
    """
    m = BUILD_PATTERN.search(Git.commit_subject())
    if m and "build_ref" in m.groupdict():
        build_ref = m.group("build_ref")
        return build_ref
    return ""


def checkout_build_commit() -> str:
    """
    Checkout commit to build wheels for
    """
    ref = get_build_ref()
    if ref:
        return Git.checkout(ref)
    return ""


def get_build_tag():
    """
    Get tag of commit to be built
    """
    return Git.get_tag_at_commit(get_build_ref())


def on_gha() -> bool:
    """
    Return True if running on Github Actions
    """
    return os.environ.get("GITHUB_ACTIONS") == "true"


def on_cirrus() -> bool:
    """
    Return True if running on Cirrus CI
    """
    return os.environ.get("CIRRUS_CI") == "true"


def skip_ci_message() -> bool:
    """
    Return True if skipping all CI
    """
    return "[skip ci]" in Git.commit_message()


def skip_gha_message() -> bool:
    """
    Return True if skipping Github Actions
    """
    return "[skip gha]" in Git.commit_message()


def skip_cirrus_message() -> bool:
    """
    Return True if skipping Github Actions
    """
    return "[skip cirrus]" in Git.commit_message()


def skip_build() -> bool:
    """
    Return True if skip
    """
    return (
        skip_ci_message()
        or (on_gha() and skip_gha_message())
        or (on_cirrus() and skip_cirrus_message())
    )


def is_wheel_build() -> bool:
    """
    Return True if to build
    """
    if skip_build():
        return False

    m = BUILD_PATTERN.search(Git.commit_message())
    if not m:
        return False

    return True


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
    Return True if wheels should be built
    """
    return (
        is_wheel_build() or
        os.environ.get("GITHUB_EVENT_NAME") == "schedule" or
        os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch"
    )


def can_release() -> bool:
    """
    Return True if tag what we expect for a release
    """
    tag = get_build_tag()
    return (
        bool(tag) and
        Git.is_annotated(tag) and
        can_build() and
        is_release_tag(tag) and
        is_release_tag_message_ok(tag)
    )


def can_pre_release() -> bool:
    """
    Return True if tag what we expect for a pre_release
    """
    tag = get_build_tag()
    return (
        bool(tag) and
        Git.is_annotated(tag) and
        can_build() and
        is_pre_release_tag(tag) and
        is_release_tag_message_ok(tag)
    )


def process_request(arg: str) -> str:
    if arg in QUESTIONS:
        result = QUESTIONS.get(arg, lambda: False)()
        if not isinstance(result, str):
            result = str(result).lower()
    else:
        result = ACTIONS.get(arg, lambda: "")()
    return result


QUESTIONS: dict[str, Ask] = {
    "can_i_build": can_build,
    "can_i_release": can_release,
    "can_i_pre_release": can_pre_release,
    "skip_build": skip_build,
    "skip_ci_message": skip_ci_message,
    "skip_gha_message": skip_gha_message,
    "is_wheel_build": is_wheel_build,
    "commit_message": Git.commit_message
}


ACTIONS: dict[str, Do] = {
    "checkout_build_commit": checkout_build_commit
}


if __name__ == "__main__":
    if len(sys.argv) == 2:
        arg = sys.argv[1]
        print(process_request(arg))
