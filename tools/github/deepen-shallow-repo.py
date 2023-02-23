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
from _repo import Repo, GithubInfo


def main():
    """
    Retrieve more commits upto version tag
    """
    repo = Repo(GithubInfo())
    n = 5
    while not repo.can_describe():
        if repo.is_shallow():
            repo.deepen(n)
            n *= 2
        else:
            break


if __name__ == "__main__":
    main()
