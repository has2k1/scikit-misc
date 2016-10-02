from os.path import join


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info, dict_append
    config = Configuration('loess', parent_package, top_path)

    # Configuration of LOESS
    f_sources = ('loessf.f', 'linpack_lite.f')
    config.add_library('floess',
                       sources=[join('src', x) for x in f_sources])
    blas_info = get_info('blas_opt')
    build_info = {}
    dict_append(build_info, **blas_info)
    dict_append(build_info, libraries=['floess'])
    sources = ['_loess.c', 'loess.c', 'loessc.c', 'misc.c', 'predict.c']
    depends = ['S.h', 'cloess.h', 'loess.h',
               '_loess.pyx',
               'c_loess.pxd']
    config.add_extension('_loess',
                         sources=[join('src', x) for x in sources],
                         depends=[join('src', x) for x in depends],
                         extra_compile_args=['-g'],
                         extra_link_args=['-g'],
                         **build_info)
    config.add_data_dir('tests')
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    config = configuration(top_path='').todict()
    setup(**config)
