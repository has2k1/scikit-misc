#!/usr/bin/env python

# This script allows `git describe` to work on github actions.
# It assumes that the repo has been checkout e.g. with
#
# jobs:
#    steps:
#       - name: Checkout
#         uses: actions/checkout@v3
#
# This fetches a single commit and results in a shallow repository.
# Here we try to fetch more commits (deepen the repo) until we reach a
# version tag. Git describe need a version tag.
from _repo import Git


def main():
    """
    Retrieve more commits upto version tag
    """
    n = 5
    while not Git.can_describe():
        if Git.is_shallow():
            Git.deepen(n)
            n *= 2
        else:
            break


if __name__ == "__main__":
    main()
