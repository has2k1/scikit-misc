# Ref: https://github.com/scipy/scipy/blob/main/scipy/meson.build
# commit:  053a3e8
#
# NOTES:
# - As scikit-misc does not use these librares/languages
#   - c++
#   - f2py (https://numpy.org/doc/stable/f2py/),
#   - pythran
#   - pybind

# Platform detection
is_mingw = is_windows and cc.get_define('__MINGW32__') != ''
if is_mingw and ff.get_id() != 'gcc'
  error('If you are using GCC on Windows, you must also use GFortran! Detected ' + ff.get_id())
endif

c_args = []
cython_c_args = ['-DCYTHON_CCOMPLEX=0'] # see gh-18975 for why we need this
cython_c_link_args = []
fortran_args = []
fortran_link_args = []

if is_mingw
  is_mingw_built_python = run_command(
    py3, ['-c', 'import sysconfig; print(sysconfig.get_platform())'],
    check: true).stdout().strip().startswith('mingw')
  if not is_mingw_built_python
    # For mingw-w64, link statically against the UCRT.
    gcc_link_args = ['-lucrt', '-static']
    add_project_link_arguments(gcc_link_args, language: ['c', 'cpp'])
    # Force gcc to float64 long doubles for compatibility with MSVC
    # builds, for C only.
    add_project_arguments('-mlong-double-64', language: 'c')
  endif
  # Make fprintf("%zd") work (see https://github.com/rgommers/scipy/issues/118)
  add_project_arguments('-D__USE_MINGW_ANSI_STDIO=1', language: ['c', 'cpp'])
endif



thread_dep = dependency('threads', required: false)

# NumPy include directory - needed in all submodules
# The chdir is needed because within numpy there's an `import signal`
# statement, and we don't want that to pick up scipy's signal module rather
# than the stdlib module. The try-except is needed because when things are
# split across drives on Windows, there is no relative path and an exception
# gets raised. There may be other such cases, so add a catch-all and switch to
# an absolute path. Relative paths are needed when for example a virtualenv is
# placed inside the source tree; Meson rejects absolute paths to places inside
# the source tree.
# For cross-compilation it is often not possible to run the Python interpreter
# in order to retrieve numpy's include directory. It can be specified in the
# cross file instead:
#   [properties]
#   numpy-include-dir = /abspath/to/host-pythons/site-packages/numpy/core/include
#
# This uses the path as is, and avoids running the interpreter.
incdir_numpy = meson.get_external_property('numpy-include-dir', 'not-given')
if incdir_numpy == 'not-given'
  incdir_numpy = run_command(py3,
    [
      '-c',
      '''import os
os.chdir(os.path.join("..", "tools"))
import numpy as np
try:
  incdir = os.path.relpath(np.get_include())
except Exception:
  incdir = np.get_include()
print(incdir)
  '''
    ],
    check: true
  ).stdout().strip()

  # We do need an absolute path to feed to `cc.find_library` below
  _incdir_numpy_abs = run_command(py3,
    ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
    check: true
  ).stdout().strip()
else
  _incdir_numpy_abs = incdir_numpy
endif
inc_np = include_directories(incdir_numpy)
# Don't use the deprecated NumPy C API. Define this to a fixed version instead of
# NPY_API_VERSION in order not to break compilation for released SciPy versions
# when NumPy introduces a new deprecation.
numpy_nodepr_api = ['-DNPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION']
np_dep = declare_dependency(include_directories: inc_np, compile_args: numpy_nodepr_api)

# We do need an absolute path to feed to `cc.find_library` below
_incdir_numpy_abs = run_command(py3,
  ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
  check: true
).stdout().strip()

npymath_path = _incdir_numpy_abs / '..' / 'lib'
npymath_lib = cc.find_library('npymath', dirs: npymath_path)
###

cython_args = ['-3', '--fast-fail', '--output-file', '@OUTPUT@', '--include-dir', '@BUILD_ROOT@', '@INPUT@']

#### Check Supported Compiler Flags
# Check if compiler flags are supported. This is necessary to ensure that SciPy
# can be built with any supported compiler. We need so many warning flags
# because we want to be able to build with `-Werror` in CI; that ensures that
# for new code we add, there are no unexpected new issues introduced.
#
# Cleaning up code so we no longer need some of these warning flags is useful,
# but not a priority.
#
# The standard convention used here is:
#   - for C, drop the leading dash and turn remaining dashes into underscores
#   - for C++, prepend `_cpp` and turn remaining dashes into underscores
#   - for Fortran, prepend `_fflags` and turn remaining dashes into underscores

# C warning flags
Wno_maybe_uninitialized = cc.get_supported_arguments('-Wno-maybe-uninitialized')
Wno_discarded_qualifiers = cc.get_supported_arguments('-Wno-discarded-qualifiers')
Wno_empty_body = cc.get_supported_arguments('-Wno-empty-body')
Wno_implicit_function_declaration = cc.get_supported_arguments('-Wno-implicit-function-declaration')
Wno_parentheses = cc.get_supported_arguments('-Wno-parentheses')
Wno_switch = cc.get_supported_arguments('-Wno-switch')
Wno_unused_label = cc.get_supported_arguments('-Wno-unused-label')
Wno_unused_result = cc.get_supported_arguments('-Wno-unused-result')
Wno_unused_variable = cc.get_supported_arguments('-Wno-unused-variable')

# Fortran warning flags
_fflag_Wno_argument_mismatch = ff.get_supported_arguments('-Wno-argument-mismatch')
_fflag_Wno_conversion = ff.get_supported_arguments('-Wno-conversion')
_fflag_Wno_intrinsic_shadow = ff.get_supported_arguments('-Wno-intrinsic-shadow')
_fflag_Wno_maybe_uninitialized = ff.get_supported_arguments('-Wno-maybe-uninitialized')
_fflag_Wno_surprising = ff.get_supported_arguments('-Wno-surprising')
_fflag_Wno_uninitialized = ff.get_supported_arguments('-Wno-uninitialized')
_fflag_Wno_unused_dummy_argument = ff.get_supported_arguments('-Wno-unused-dummy-argument')
_fflag_Wno_unused_label = ff.get_supported_arguments('-Wno-unused-label')
_fflag_Wno_unused_variable = ff.get_supported_arguments('-Wno-unused-variable')
_fflag_Wno_tabs = ff.get_supported_arguments('-Wno-tabs')
# The default list of warnings to ignore from Fortran code. There is a lot of
# old, vendored code that is very bad and we want to compile it silently (at
# least with GCC and Clang)
fortran_ignore_warnings = ff.get_supported_arguments(
 _fflag_Wno_argument_mismatch,
 _fflag_Wno_conversion,
 _fflag_Wno_maybe_uninitialized,
 _fflag_Wno_unused_dummy_argument,
 _fflag_Wno_unused_label,
 _fflag_Wno_unused_variable,
 _fflag_Wno_tabs,
)

# Deal with M_PI & friends; add `use_math_defines` to c_args or cpp_args
# Cython doesn't always get this right itself (see, e.g., gh-16800), so
# explicitly add the define as a compiler flag for Cython-generated code.
use_math_defines = []
if is_windows
  use_math_defines += ['-D_USE_MATH_DEFINES']
  # cgwin gfortran may insert symbol/function ___chkstk_ms to pad space around large
  # arrays allocated on the stack. But to the linker (msvsc) that symbol will be
  # undefined. These prevent gfortran from doing any stack checks and inserting that
  # symbols. An alternative would have been to add libgfortran as a link argument,
  # but what we tried failed.
  fortran_args += [
    ff.get_supported_arguments('-fno-stack-check'),
    ff.get_supported_arguments('-fno-stack-protector'),
    ff.get_supported_arguments('-mno-stack-arg-probe'),
  ]
elif is_macos
  Wlno_compact_unwind = cc.get_supported_link_arguments('-Wl,-no_compact_unwind')
  # TODO: This should not exist long term
  # Disable warning about undefined dynamic lookup of symbols. It is harmless.
  # https://github.com/python/cpython/issues/97524#issuecomment-1270616173
  Wlno_fixup_chains = cc.get_supported_link_arguments('-Wl,-no_fixup_chains')
  cython_c_link_args += [
    Wlno_compact_unwind,
    Wlno_fixup_chains
  ]
endif

cython_c_args += [use_math_defines]
###

#### Compilers
compilers = {
  'C': cc,
  'CYTHON': meson.get_compiler('cython'),
  'FORTRAN': meson.get_compiler('fortran')
}

machines = {
  'HOST': host_machine,
  'BUILD': build_machine,
}


#### Configuration Information
conf_data = configuration_data()
conf_data.set('VERSION', meson.project_version())

# Set compiler information
foreach name, compiler : compilers
  conf_data.set(name + '_COMP', compiler.get_id())
  conf_data.set(name + '_COMP_LINKER_ID', compiler.get_linker_id())
  conf_data.set(name + '_COMP_VERSION', compiler.version())
  conf_data.set(name + '_COMP_CMD_ARRAY', ', '.join(compiler.cmd_array()))
  conf_data.set(name + '_COMP_ARGS', ', '.join(
      get_option(name.to_lower() + '_args')
    )
  )
  conf_data.set(name + '_COMP_LINK_ARGS', ', '.join(
      get_option(name.to_lower() + '_link_args')
    )
  )
endforeach

# Machines CPU and system information
foreach name, machine : machines
  conf_data.set(name + '_CPU', machine.cpu())
  conf_data.set(name + '_CPU_FAMILY', machine.cpu_family())
  conf_data.set(name + '_CPU_ENDIAN', machine.endian())
  conf_data.set(name + '_CPU_SYSTEM', machine.system())
endforeach

conf_data.set('CROSS_COMPILED', meson.is_cross_build())

# Python information
conf_data.set('PYTHON_PATH', py3.full_path())
conf_data.set('PYTHON_VERSION', py3.language_version())
###

skmisc_dir = py3.get_install_dir() / 'skmisc'

#### Include Python Sources in this Directory
# Copy the main __init__ to the build dir
python_sources = [
  '__init__.py',
  '_distributor_init.py',
]

py3.install_sources(
  python_sources,
  subdir: 'skmisc'
)

#### Build __config__.py
configure_file(
  input: '__config__.py.in',
  output: '__config__.py',
  configuration : conf_data,
  install_dir: skmisc_dir,
)
###

#### Build _version.py
configure_file(
  input: '_version.py.in',
  output: '_version.py',
  configuration: conf_data,
  install_dir: skmisc_dir,
)

# Meson does not inculde any generated files in the source distribution
# We want the generated _version.py so that a versioned build can be
# created from the source dist.
meson.add_dist_script('_build_utils/copy_version_to_dist.py')

###

#### Included sub-packages
subdir('loess')
###
