#### Project Information
project(
  'scikit-misc',
  'c', 'cython',
  version: run_command('skmisc/_build_utils/version_please.py', check: true).stdout().strip(),
  license: 'BSD-3',
  meson_version: '>= 1.1.0',
  default_options: [
    'buildtype=debugoptimized',
    'c_std=c99',
    'fortran_std=legacy',
    'blas=openblas',
    'lapack=openblas'
  ],
)

# Adding at project level causes many spurious -lgfortran flags.
add_languages('fortran', native: false)
###

#### Configuration Information
conf_data = configuration_data()

# The object above is stored so that we can write it to
# a file.
conf_data.set('VERSION', meson.project_version())
###

#### Setup Python
# https://mesonbuild.com/Python-module.html
py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

skmisc_dir = py.get_install_dir() / 'skmisc'

conf_data.set('PYTHON_PATH', py.full_path())
conf_data.set('PYTHON_VERSION', py.language_version())
###

#### Setup Compilers
cc = meson.get_compiler('c')
ff = meson.get_compiler('fortran')
# generator() doesn't accept compilers, only found programs. Cast it.
cy = meson.get_compiler('cython')
cython = find_program(cy.cmd_array()[0])

compilers = {
  'C': cc,
  'CYTHON': cy,
  'FORTRAN': ff
}

foreach name, compiler : compilers
  # conf_data.set(name + '_COMP_CMD_ARRAY', compiler.cmd_array())
  conf_data.set(name + '_COMP_CMD_ARRAY', compiler.get_id())
  conf_data.set(name + '_COMP', compiler.get_id())
  conf_data.set(name + '_COMP_LINKER_ID', compiler.get_linker_id())
  conf_data.set(name + '_COMP_VERSION', compiler.version())
  conf_data.set(name + '_COMP_CMD_ARRAY', ', '.join(compiler.cmd_array()))
endforeach
###

#### Setup Machine CPU & System Information
machines = {
  'HOST': host_machine,
  'BUILD': build_machine,
}

foreach name, machine : machines
  conf_data.set(name + '_CPU', machine.cpu())
  conf_data.set(name + '_CPU_FAMILY', machine.cpu_family())
  conf_data.set(name + '_CPU_ENDIAN', machine.endian())
  conf_data.set(name + '_CPU_SYSTEM', machine.system())
endforeach

conf_data.set('CROSS_COMPILED', meson.is_cross_build())
###

#### Compiler Arguments and Settings
# Check compiler is recent enough (see "Toolchain Roadmap" for details)
if cc.get_id() == 'gcc'
  if not cc.version().version_compare('>=8.0')
    error('scikit-misc requires GCC >= 8.0')
  endif
elif cc.get_id() == 'msvc'
  if not cc.version().version_compare('>=19.20')
    error('scikit-misc requires at least vc142 (default with Visual Studio 2019) ' + \
          'when building with MSVC')
  endif
endif

# link against the standard math library since we use it in c
m_dep = cc.find_library('m', required : false)
if m_dep.found()
  add_project_link_arguments('-lm', language : 'c')
endif

add_languages('fortran', native: false)
if ff.has_argument('-Wno-conversion')
  add_project_arguments('-Wno-conversion', language: 'fortran')
endif


is_windows = host_machine.system() == 'windows'
is_macos = host_machine.system() == 'darwin'

# Intel compilers default to fast-math, so disable it if we detect Intel
# compilers. A word of warning: this may not work with the conda-forge
# compilers, because those have the annoying habit of including lots of flags
# that are gcc-specific in CFLAGS/CXXFLAGS/FFLAGS, which throws off the
# detection logic below. You have to remove the wrong flags (only `-isystem`
# is actually needed, everything else shouldn't be there).
_intel_cflags = []
_intel_fflags = []
if cc.get_id() == 'intel'
  _intel_cflags += cc.get_supported_arguments('-fp-model=strict')
elif cc.get_id() == 'intel-cl'
  _intel_cflags += cc.get_supported_arguments('/fp:strict')
endif
if ff.get_id() == 'intel'
  _intel_fflags = ff.get_supported_arguments('-fp-model=strict')
  minus0_arg = ['-assume', 'minus0']
  if ff.has_multi_arguments(minus0_arg)
    _intel_fflags += minus0_arg
  endif
elif ff.get_id() == 'intel-cl'
  # Intel Fortran on Windows does things differently, so deal with that
  # (also specify dynamic linking and the right name mangling)
  _intel_fflags = ff.get_supported_arguments(
    '/fp:strict', '/MD', '/names:lowercase', '/assume:underscore',
    '/assume:minus0'
  )
endif
add_project_arguments(_intel_cflags, language: ['c'])
add_project_arguments(_intel_fflags, language: 'fortran')

cython_c_args = []
cython_args = ['-3', '--fast-fail', '--output-file', '@OUTPUT@', '--include-dir', '@BUILD_ROOT@', '@INPUT@']
cython_c_link_args = []
###

#### Include Main Package
subdir('skmisc')
###
