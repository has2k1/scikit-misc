fs = import('fs')

__init__py = fs.copyfile('__init__.py')

#### pyx -> c generator, depending on copied pxd files and init
loess_cython_gen = generator(
  cython,
  arguments: cython_args,
  output: '_@BASENAME@.c',
  depends: [
    __init__py,
    fs.copyfile('src/c_loess.pxd'),
    fs.copyfile('src/loess.h')
  ]
)
###

#### Fortran Loess into a static library
floess_sources = [
  'src/loessf.f',
  'src/linpack_lite.f'
]

floess_lib = static_library(
  'floess',
  floess_sources,
  fortran_args: [fortran_ignore_warnings],
  dependencies: [cblas, blas_dep, lapack_dep],
  gnu_symbol_visibility: 'hidden',
)
###

#### Loess Extenstion Module
py3.extension_module(
  '_loess',
   sources: [
     'src/loess.c',
     'src/loessc.c',
     'src/misc.c',
     'src/predict.c',
     loess_cython_gen.process('src/_loess.pyx')
   ],
   c_args: [cython_c_args],
   dependencies: [py3_dep, npymath_lib, np_dep],
   link_with: [floess_lib],
   link_language: 'c',
   install: true,
   subdir: 'skmisc/loess'
)
###

#### Include Python Sources in this Directory
# Copy the subpackage __init__ to the build dir
python_sources = [
  '__init__.py'
]

py3.install_sources(
  python_sources,
  subdir: 'skmisc/loess'
)
###

#### Included sub-packages
subdir('tests')
###
