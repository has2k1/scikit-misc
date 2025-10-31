#### Project Information
project(
  'scikit-misc',
  'c', 'cython',
  version: run_command('skmisc/_build_utils/version_please.py', check: true).stdout().strip(),
  license: 'BSD-3',
  meson_version: '>= 1.5.0',
  default_options: [
    'buildtype=debugoptimized',
    'c_std=c17',
    'fortran_std=legacy',
    'pkgconfig.relocatable=true',
  ],
)

###

#### Setup Python
# https://mesonbuild.com/Python-module.html
py3 = import('python').find_installation(pure: false)
py3_dep = py3.dependency()
###

#### Define Compilers
cc = meson.get_compiler('c')
cy = meson.get_compiler('cython')
# generator() doesn't accept compilers, only found programs. Cast it.
cython = find_program(cy.cmd_array()[0])
###

#### Compiler Arguments and Settings
# Check compiler is recent enough (see "Toolchain Roadmap" for details)
if cc.get_id() == 'gcc'
  if not cc.version().version_compare('>=9.1')
    error('scikit-misc requires GCC >= 9.1')
  endif
elif cc.get_id() == 'clang' or cc.get_id() == 'clang-cl'
  if not cc.version().version_compare('>=15.0')
    error('scikit-misc requires clang >= 15.0')
  endif
elif cc.get_id() == 'msvc'
  if not cc.version().version_compare('>=19.20')
    error('scikit-misc requires at least vc142 (default with Visual Studio 2019) ' + \
          'when building with MSVC')
  endif
endif

if not cy.version().version_compare('>=3.0.8')
  error('NumPy requires Cython >= 3.0.8')
endif

# Add default compile flags for any compiler that supports them.
# Note that MSVC does not support strict aliasing at all, and neither do the
# Intel compilers on Windows, so the `-fno` flavor of the flag should be fine.
add_project_arguments(
  cc.get_supported_arguments( '-fno-strict-aliasing'), language : 'c'
)

# link against the standard math library since we use it in c
m_dep = cc.find_library('m', required : false)
if m_dep.found()
  add_project_link_arguments('-lm', language : 'c')
endif

# Adding at project level causes many spurious -lgfortran flags.
add_languages('fortran', native: false)
ff = meson.get_compiler('fortran')
if ff.has_argument('-Wno-conversion')
  add_project_arguments('-Wno-conversion', language: 'fortran')
endif

is_windows = host_machine.system() == 'windows'
is_macos = host_machine.system() == 'darwin'

if is_macos
  if cc.has_link_argument('-Wl,-dead_strip')
    # Allow linker to strip unused symbols
    add_project_link_arguments('-Wl,-dead_strip', language : ['c', 'fortran'])
  endif
endif

# Intel compilers default to fast-math, so disable it if we detect Intel
# compilers. A word of warning: this may not work with the conda-forge
# compilers, because those have the annoying habit of including lots of flags
# that are gcc-specific in CFLAGS/CXXFLAGS/FFLAGS, which throws off the
# detection logic below. You have to remove the wrong flags (only `-isystem`
# is actually needed, everything else shouldn't be there).
_intel_cflags = []
_intel_fflags = []
if cc.get_id() in ['intel', 'intel-llvm']
  _intel_cflags += cc.get_supported_arguments('-fp-model=strict')
elif cc.get_id() in ['intel-cl', 'intel-llvm-cl']
  _intel_cflags += cc.get_supported_arguments('/fp:strict')
endif
if ff.get_id() in ['intel', 'intel-llvm']
  _intel_fflags = ff.get_supported_arguments('-fp-model=strict')
  minus0_arg = ['-assume', 'minus0']
  if ff.has_multi_arguments(minus0_arg)
    _intel_fflags += minus0_arg
  endif
elif ff.get_id() in ['intel-cl', 'intel-llvm-cl']
  # Intel Fortran on Windows does things differently, so deal with that
  # (also specify dynamic linking and the right name mangling)
  _intel_fflags = ff.get_supported_arguments(
    '/fp:strict', '/MD', '/names:lowercase', '/assume:underscore',
    '/assume:minus0'
  )
endif
add_project_arguments(_intel_cflags, language: ['c'])
add_project_arguments(_intel_fflags, language: 'fortran')
###

#### Include Main Package
subdir('skmisc')
###
