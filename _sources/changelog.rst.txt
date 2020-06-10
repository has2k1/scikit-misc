Changelog
=========

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
