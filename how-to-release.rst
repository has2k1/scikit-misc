##############
How to release
##############

Testing
=======

* `cd` to the root of project and run
  ::

    make test

* Or, to test in all environments
  ::

    tox

* Once all the tests pass move on


Tagging
=======

Check out the master branch, tag with the version number & push the tags

  ::

    git checkout master
    git tag -a v0.1.0 -m 'Version: 0.1.0'
    git push upstream --tags

Note the `v` before the version number.


Build Wheels
============
Clone/cd into the wheels repository, edit `.github/workflows/wheels.yml` to point
to the version. i.e.

  ::

    git clone https://github.com/has2k1/scikit-misc-wheels  # (optional)
    cd scikit-misc-wheels
    git submodule foreach 'git fetch --all; git reset --hard origin/master'

    # Edit .github/workflows/wheels.yml and set the version e.g.
    #     - BUILD_COMMIT: v0.1.0

    git commit -a -m 'Version: 0.1.0'
    git push origin --tags

Check `Github Actions <https://github.com/has2k1/scikit-misc-wheels/actions>`_ to confirm
that all the builds for the last commit pass. Debug as necessary, then continue
below.


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
