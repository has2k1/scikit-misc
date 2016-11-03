

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('skmisc', parent_package, top_path)
    config.add_subpackage('loess')
    config.make_config_py()  # installs __config__.py
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
