#!/bin/bash

# This file should be sourced by release.sh which defines
# the following environment variables
#   - BUILD_DIR
#   - VERSION

# The wheels are placed in $BUILD_DIR/dist
# That is where the upload command expects them to be.

temp_dir=`mktemp -d`
pushd $temp_dir

git clone --depth=10 https://github.com/has2k1/scikit-misc-wheels
git checkout wheelhouse

mkdir -p "$BUILD_DIR/dist"
cp -a "$VERSION/." "$BUILD_DIR/dist"

popd
rm -rf $temp_dir
