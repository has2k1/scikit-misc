"""
=================================================
Locally-weighted regression (:mod:`skmisc.loess`)
=================================================

Loess is a procedure for estimating a regression surface by a multivariate
smoothing procedure. A linear or quadratic function of the independent
variables is fit in a moving fashion that is analogous to how a moving
average is computed for a time series.

Compared to approaches that fit global parametric functions, loess
substantially increases the domain of surfaces that can be estimated without
distortion. However, analogues of the statistical procedures used
in parametric function fitting -- for example, ANOVA and t intervals --
involve statistics whose distributions are well approximated by familiar
distributions.

.. autosummary::
   :toctree: generated/
   :template: cython_class.rst

   loess
   loess_inputs
   loess_model
   loess_control
   loess_outputs
   loess_prediction
   loess_confidence_intervals
   loess_anova

Source
------
The original source code was written by William S. Cleveland, Eric Grosse
and Ming-Jen Shyu. It is available at http://www.netlib.org/a/dloess. It
was initially adapted to for use in Scipy by Pierre GF Gerard-Marchant.

For more see references [1]_ [2]_ and [3]_.

.. [1] W. S. Cleveland, E. Grosse, and M. J. Shyu. Local Regression Models.
   In J. M. Chambers and T. Hastie, editors, Statistical Models in S,
   pages 309--376. Chapman and Hall, New York, 1992.

.. [2] W. S. Cleveland, S. J. Devlin, and E. Grosse. Regression by Local
   Fitting: Methods, Properties, and Computing. Journal of Econometrics, 37:
   pp. 87--114. 1988.

.. [3] W. S. Cleveland. Robust Locally Weighted Regression and Smoothing
   Scatterplots. Journal of the American Statistical Association, 74:
   pp. 829--836. 1979.
"""
from ._loess import (loess, loess_model, loess_inputs, loess_control,
                     loess_outputs, loess_prediction,
                     loess_confidence_intervals, loess_anova)


__all__ = ['loess', 'loess_model', 'loess_control', 'loess_inputs',
           'loess_model', 'loess_outputs', 'loess_prediction',
           'loess_confidence_intervals', 'loess_anova']
