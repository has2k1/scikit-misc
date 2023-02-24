import os
import shutil
import sys
from pathlib import Path

import click
from devpy import util

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

    site_path = util.get_site_packages(build_dir)
    if site_path is None:
        print("No built scikit-misc found; run `./dev.py build` first.")
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
    site_path = util.get_site_packages(build_dir)
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
    site_path = util.get_site_packages(build_dir)
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
    # meson directory. It creates an sdist with PKG-INFO
    util.run([
        "python",
        "-m",
        "build",
        "--no-isolation",
        "--skip-dependency-check",
        "--sdist",
        ".",
    ], replace=True)
