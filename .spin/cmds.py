import os
import shutil
import sys
from pathlib import Path

import click
from spin import util
from spin.cmds import meson


@click.command()
@click.option(
    "--build-dir",
    default="build",
    help="Build directory; default is `$PWD/build`"
)
@click.option(
    "--clean",
    is_flag=True,
    help="Clean previously built docs before building"
)
def docs(build_dir, clean=False):
    """
    📖 Build documentation
    """
    if clean:
        doc_dir = Path("./doc/build").absolute()
        if doc_dir.is_dir():
            print(f"Removing `{doc_dir}`")
            shutil.rmtree(doc_dir)

    site_path = meson._get_site_packages()
    if site_path is None:
        print("No built scikit-misc found; run `./spin build` first.")
        sys.exit(1)

    util.run(["pip", "install", "-q", "-r", "requirements/docs.txt"])

    PYTHONPATH = os.environ.get("PYTHONPATH", "")
    os.environ["SPHINXOPTS"] = "-W"
    os.environ['PYTHONPATH'] = f'{site_path}{os.sep}:{PYTHONPATH}'
    util.run(["make", "-C", "doc", "html"], replace=True)


@click.command()
@click.option(
    "--build-dir",
    default="build",
    help="Build directory; default is `$PWD/build`"
)
def coverage(build_dir):
    """
    📊 Generate coverage report
    """
    site_path = meson._get_site_packages()
    util.run([
        "python",
        "-m",
        "coverage",
        "report",
        "--data-file",
        f"{site_path}/.coverage",
    ], replace=True)


@click.command()
@click.option(
    "--build-dir",
    default="build",
    help="Build directory; default is `$PWD/build`"
)
def coverage_html(build_dir):
    """
    📊 Generate HTML coverage report
    """
    site_path = meson._get_site_packages()
    util.run([
        "python",
        "-m",
        "coverage",
        "html",
        "--data-file",
        f"{site_path}/.coverage",
    ], replace=True)


@click.command()
def sdist():
    """
    📦 Build a source distribution in `build/meson-dist/`.
    """
    # Using the build module gives better results than using
    # meson directly. It creates an sdist with PKG-INFO
    util.run([
        "python",
        "-m",
        "build",
        "--no-isolation",
        "--skip-dependency-check",
        "--sdist",
        ".",
    ], replace=True)

@click.command(context_settings={
    'ignore_unknown_options': True
})
@click.option(
    "--with-scipy-openblas", type=click.Choice(["32", "64"]),
    default=None, required=True,
    help="Build with pre-installed scipy-openblas32 or scipy-openblas64 wheel"
)
def config_openblas(with_scipy_openblas):
    """🔧 Create .openblas/scipy-openblas.pc file

    Also create _distributor_init_local.py

    Requires a pre-installed scipy-openblas64 or scipy-openblas32
    """
    _config_openblas(with_scipy_openblas)


# from numpy
@click.option(
    "--with-scipy-openblas", type=click.Choice(["32", "64"]),
    default=None,
    help="Build with pre-installed scipy-openblas32 or scipy-openblas64 wheel"
)
@click.option(
    "--clean", is_flag=True,
    help="Clean build directory before build"
)
@click.option(
    "-v", "--verbose", is_flag=True,
    help="Print all build output, even installation"
)
@click.argument("meson_args", nargs=-1)
@click.pass_context
def build(ctx, meson_args, with_scipy_openblas, jobs=None, clean=False, verbose=False, quiet=False, *args, **kwargs):
    # XXX keep in sync with upstream build
    if with_scipy_openblas:
        _config_openblas(with_scipy_openblas)
    ctx.params.pop("with_scipy_openblas", None)
    ctx.forward(meson.build)


def _config_openblas(blas_variant):
    import importlib
    basedir = os.getcwd()
    openblas_dir = os.path.join(basedir, ".openblas")
    pkg_config_fname = os.path.join(openblas_dir, "scipy-openblas.pc")
    if blas_variant:
        module_name = f"scipy_openblas{blas_variant}"
        try:
            openblas = importlib.import_module(module_name)
        except ModuleNotFoundError:
            raise RuntimeError(f"'pip install {module_name} first")
        local = os.path.join(basedir, "skmisc", "_distributor_init_local.py")
        with open(local, "wt", encoding="utf8") as fid:
            fid.write(f"import {module_name}\n")
        os.makedirs(openblas_dir, exist_ok=True)
        with open(pkg_config_fname, "wt", encoding="utf8") as fid:
            fid.write(
                openblas.get_pkg_config(use_preloading=True)
            )
