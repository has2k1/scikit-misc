Changelog
=========

v0.5.2
------
*(2025-10-31)*

- Bumped minimum requred numpy to v1.26.4.

v0.5.1
------
*(2024-07-25)*

- Fixed 1.x

v0.5.0
------
*(2024-07-25)*

- Fixed support for numpy 2.0

v0.4.0
------
*(2024-06-22)*

- Dropped support for python==3.9
- Made compatible with numpy 2.0
- Include tools/ in the source distribution

v0.3.1
------
*(2023-12-14)*

- Include tools/ in the source distribution

v0.3.0
------
*(2023-07-06)*

- Added support for pickling :class:`~skmisc.loess.loess` objects.

v0.2.0
------
*(2023-04-15)*

- Requires numpy>=1.19.5
- Changed the build system to meson-build
- This is the first release that has wheels for WindowsOS

v0.1.4
------
*(2021-05-05)*

- This is a maintenance release to keep up to date with deprecations
  in upstream packages.


v0.1.3
------
*(2020-05-14)*

- Requires numpy>=1.14.5

v0.1.2
------
*(2020-05-10)*

- Fixed :meth:`~skmisc.loess.loess_output.summary` so that it
  does not result in an exception.

- Fixed :meth:`~skmisc.loess.loess_prediction.values` (and all other
  attributes/methods that return arrays) so that they do not show
  corrupted results. (:issue:`8`)

v0.1.1
------
*(2017-04-12)*

- Python 3.6

- Added memory allocation checks for `malloc`.

- Added an example of how to use :class:`~skmisc.loess.loess`

- Fixed misplaced documentation strings and added documentation to
  the loess properties.

v0.1.0
------
*(2016-11-08)*

First public release
