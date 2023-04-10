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

from _repo import Git, Workspace


Ask: TypeAlias = Callable[[], bool]
Do: TypeAlias = Callable[[], str]

WS = Workspace()

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
    r"\[wheel build(: (?P<build_ref>.+?))?\]$"
)

BUILD_PATTERN_CI = re.compile(
    r"\[wheel build - (?P<ci>GHA|cirrus)\]$"
)

def checkout_build_commit() -> str:
    """
    Checkout commit to build wheels for
    """
    m = BUILD_PATTERN.search(Git.commit_subject())
    if m and "build_ref" in m.groupdict():
        build_ref = m.group("build_ref")
        return Git.checkout(build_ref)
    return ""


def is_wheel_build_for_ci() -> bool:
    """
    Return True if to build only on a specific ci"
    """
    m = BUILD_PATTERN_CI.search(Git.commit_subject())
    if not m:
        return False

    ci = m.group("ci").lower()
    gha = os.environ.get("GITHUB_ACTIONS") == "true"
    cirrus = os.environ.get("CIRRUS_CI") == "true"
    return (cirrus and ci == "cirrus") or (gha and ci == "gha")


def is_wheel_build_for_all() -> bool:
    """
    Return True commit subject ends in "[wheel build]"
    """
    return bool(BUILD_PATTERN.search(Git.commit_subject()))


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
        is_wheel_build_for_all() or
        is_wheel_build_for_ci() or
        can_release() or
        can_pre_release() or
        WS.event_name == "schedule" or
        WS.event_name == "workflow_dispatch"
    )


def can_release() -> bool:
    """
    Return True if tag what we expect for a release
    """
    tag = Git.head_tag()
    return (
        bool(tag) and
        WS.is_push_event() and
        Git.is_annotated(tag) and
        is_release_tag(tag) and
        is_release_tag_message_ok(tag)
    )


def can_pre_release() -> bool:
    """
    Return True if tag what we expect for a pre_release
    """
    tag = Git.head_tag()
    return (
        bool(tag) and
        WS.is_push_event() and
        Git.is_annotated(tag) and
        is_pre_release_tag(tag) and
        is_release_tag_message_ok(tag)
    )


def process_request(arg: str) -> str:
    if arg in QUESTIONS:
        result = QUESTIONS.get(arg, lambda: False)()
        result = str(result).lower()
    else:
        result = ACTIONS.get(arg, lambda: "")()
    return result


QUESTIONS: dict[str, Ask] = {
    "can_i_build": can_build,
    "can_i_release": can_release,
    "can_i_pre_release": can_pre_release,
}


ACTIONS: dict[str, Do] = {
    "checkout_build_commit": checkout_build_commit
}


if __name__ == "__main__":
    if len(sys.argv) == 2:
        arg = sys.argv[1]
        print(process_request(arg))
