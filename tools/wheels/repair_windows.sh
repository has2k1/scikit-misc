set -xe

WHEEL="$1"
DEST_DIR="$2"

# create a temporary directory in the destination folder and unpack the wheel
# into there

pushd "$DEST_DIR"
mkdir -p tmp
pushd tmp
wheel unpack "$WHEEL"
pushd scikit_misc*

# To avoid DLL hell, the file name of libopenblas that's being vendored with
# the wheel has to be name-mangled. delvewheel is unable to name-mangle PYD
# containing extra data at the end of the binary, which frequently occurs when
# building with mingw.
# We therefore find each PYD in the directory structure and strip them.

for f in $(find ./skmisc* -name '*.pyd'); do strip $f; done


# now repack the wheel and overwrite the original
wheel pack .
mv -fv ./*.whl "$WHEEL"

cd "$DEST_DIR"
rm -rf tmp

delvewheel repair -w $DEST_DIR $WHEEL
