#!/bin/bash

# This file should be sourced by release.sh which defines
# the following environment variables
#   - BUILD_DIR
#   - VERSION

# The wheels are placed in $BUILD_DIR/dist
# That is where the upload command expects them to be.

# Get replace 'alpha' and 'beta' with 'a' and 'b' respectively,
# this matches the versions created for the source and wheel
# distributions
normalize_version() {
   local version=$1

   # single character pre-release name
   version=$(echo $version | sed -e 's/\(alpha\)\([0-9]\+\)$/a\2/g')
   version=$(echo $version | sed -e 's/\(beta\)\([0-9]\+\)$/b\2/g')
   version=$(echo $version | sed -e 's/\(alpha\)$/a0/g')
   version=$(echo $version | sed -e 's/\(beta\)$/b0/g')

   # strip-off the v if any
   if [[ ${version:0:1} == 'v' ]]; then
      version=${version:1}
   fi

   echo $version
}

version=$(normalize_version $VERSION)

temp_dir=`mktemp -d`
pushd $temp_dir

git clone --depth=5 --branch=wheelhouse \
   https://github.com/has2k1/scikit-misc-wheels
cd scikit-misc-wheels

mkdir -p "$BUILD_DIR/dist"
cp -a "$version/." "$BUILD_DIR/dist"

popd
rm -rf $temp_dir
