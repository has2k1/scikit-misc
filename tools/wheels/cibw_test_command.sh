set -xe

if [[ $RUNNER_OS == "Windows" ]]; then
    # GH 20391
    PY_DIR=$(python -c "import sys; print(sys.prefix)")
    mkdir $PY_DIR/libs
fi

if [[ $RUNNER_OS == "macOS"  && $RUNNER_ARCH == "X64" ]]; then
  # Not clear why this is needed but it seems on x86_64 this is not the default
  # and without it f2py tests fail
  export DYLD_LIBRARY_PATH="$DYLD_LIBRARY_PATH:/usr/local/lib"
  # Needed so gfortran (not clang) can find system libraries like libm (-lm)
  # in f2py tests
  export LIBRARY_PATH="$LIBRARY_PATH:/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib"
fi

python -c "import sys; import skmisc; sys.exit(skmisc.test())"
