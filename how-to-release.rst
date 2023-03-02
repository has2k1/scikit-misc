##############
How to release
##############

Testing
=======

* Create a new virtual environment

* Install project, build and test
  ::

    git switch main
    pip install -r requirements/build.txt
    make build
    make test

* Test that the wheels will build on github actions

  ::

    git switch main
    git log --oneline wheels -n 20
    git switch --force-create wheels
    git commit --amend -m "TST: Preparing for release [wheel build]"
    git push origin

  1. Switch to main branch
  2. Make sure the wheel branch does not have unmerged commits
  3. Create a new wheels branch at the head of main
  4. Update the commit header to one that will get the wheels to build
  5. Push to build the while

* Once all the tests pass move on

Changelog
=========

* Modify changelog doc/changelog.rst and commit

  1. Set the version number
  2. Set the release date

Tagging
=======

Check out the main branch, tag with the version number & push the tags

  ::

    git checkout main
    # git tag -a v0.1.0a1 -m 'Version: 0.1.0a1' pre-release
    git tag -a v0.1.0 -m 'Version: 0.1.0'

The version tag for a release must be of the form `v<semantic-version>` and
the version comment of the form `Version: <semantic-version>`, as shown above.

The version tag for a test release must be of the form
`v<semantic-version>(a|b|alpha|beta)\d+` and the version comment of the form
`Version: <semantic-version>(a|b|alpha|beta)\d+`, as shown above. The pre-release
is upload to `PyPiTest <https://test.pypi.org/project/scikit-misc>`. The link is

A link to install

Build Wheels and Release
========================

Push to release

::
    git push origin --tags


Release
=======

* Make sure your `.pypirc` file is setup
  `correctly <http://docs.python.org/2/distutils/packageindex.html>`_.
  ::

    cat ~/.pypirc

* Release

  ::

    make release

Documentation
=============

When a release is tagged and pushed to github, a Github Action builds the
[documentation](https://has2k1.github.io/scikit-misc/) and automatically
pushes it to the [gh-pages](https://github.com/has2k1/scikit-misc/tree/gh-pages) branch.
See `tools/deploy_documentation.sh` for how it happens.
