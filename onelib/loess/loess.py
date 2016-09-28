# pylint: disable-msg=E1101
from __future__ import absolute_import

"""
Wrapper to loess

LOESS:
Initial C/Fortran package avialable at
http://netlib.bell-labs.com/netlib/a/dloess.gz
Initial authors: W. S. Cleveland, E. Grosse and Shyu
Adaptation to Pyrex/Python by Pierre Gerard-Marchant, 2007/03

:author: Pierre GF Gerard-Marchant
:contact: pierregm_at_uga_edu
:date: $Date$
:version: $Id$
"""
__author__ = "Pierre GF Gerard-Marchant ($Author$)"
__version__ = '1.0'
__revision__ = "$Revision$"
__date__     = '$Date$'

from . import _loess


#####---------------------------------------------------------------------------
#--- --- Loess ---
#####---------------------------------------------------------------------------
loess = _loess.loess
"""
loess : locally weighted estimates. Multi-variate version

:Keywords:
    x : ndarray
        A (n,p) ndarray of independent variables, with n the number of observations
        and p the number of variables.
    y : ndarray
        A (n,) ndarray of observations
    weights : ndarray
        A (n,) ndarray of weights to be given to individual observations in the
        sum of squared residuals that forms the local fitting criterion. If not
        None, the weights should be non negative. If the different observations
        have non-equal variances, the weights should be inversely proportional
        to the variances.
        By default, an unweighted fit is carried out (all the weights are one).
    surface : string ["interpolate"]
        Determines whether the fitted surface is computed directly at all points
        ("direct") or whether an interpolation method is used ("interpolate").
        The default ("interpolate") is what most users should use unless special
        circumstances warrant.
    statistics : string ["approximate"]
        Determines whether the statistical quantities are computed exactly
        ("exact") or approximately ("approximate"). "exact" should only be used
        for testing the approximation in statistical development and is not meant
        for routine usage because computation time can be horrendous.
    trace_hat : string ["wait.to.decide"]
        Determines how the trace of the hat matrix should be computed. The hat
        matrix is used in the computation of the statistical quantities.
        If "exact", an exact computation is done; this could be slow when the
        number of observations n becomes large. If "wait.to.decide" is selected,
        then a default is "exact" for n < 500 and "approximate" otherwise.
        This option is only useful when the fitted surface is interpolated. If
        surface is "exact", an exact computation is always done for the trace.
        Setting trace_hat to "approximate" for large dataset will substantially
        reduce the computation time.
    iterations : integer
        Number of iterations of the robust fitting method. If the family is
        "gaussian", the number of iterations is set to 0.
    cell : integer
        Maximum cell size of the kd-tree. Suppose k = floor(n*cell*span),
        where n is the number of observations, and span the smoothing parameter.
        Then, a cell is further divided if the number of observations within it
        is greater than or equal to k. This option is only used if the surface
        is interpolated.
    span : float [0.75]
        Smoothing factor, as a fraction of the number of points to take into
        account.
    degree : integer [2]
        Overall degree of locally-fitted polynomial. 1 is locally-linear
        fitting and 2 is locally-quadratic fitting.  Degree should be 2 at most.
    normalize : boolean [True]
        Determines whether the independent variables should be normalized.
        If True, the normalization is performed by setting the 10% trimmed
        standard deviation to one. If False, no normalization is carried out.
        This option is only useful for more than one variable. For spatial
        coordinates predictors or variables with a common scale, it should be
        set to False.
    family : string ["gaussian"]
        Determines the assumed distribution of the errors. The values are
        "gaussian" or "symmetric". If "gaussian" is selected, the fit is
        performed with least-squares. If "symmetric" is selected, the fit
        is performed robustly by redescending M-estimators.
    parametric_flags : sequence [ [False]*p ]
        Indicates which independent variables should be conditionally-parametric
       (if there are two or more independent variables). The argument should
       be a sequence of booleans, with the same size as the number of independent
       variables, specified in the order of the predictor group ordered in x.
    drop_square : sequence [ [False]* p]
        When there are two or more independent variables and when a 2nd order
        polynomial is used, "drop_square_flags" specifies those numeric predictors
        whose squares should be dropped from the set of fitting variables.
        The method of specification is the same as for parametric.

:Outputs:
    fitted_values : ndarray
        The (n,) ndarray of fitted values.
    fitted_residuals : ndarray
        The (n,) ndarray of fitted residuals (observations - fitted values).
    enp : float
        Equivalent number of parameters.
    s : float
        Estimate of the scale of residuals.
    one_delta: float
        Statistical parameter used in the computation of standard errors.
    two_delta : float
        Statistical parameter used in the computation of standard errors.
    pseudovalues : ndarray
        The (n,) ndarray of adjusted values of the response when robust estimation
        is used.
    trace_hat : float
        Trace of the operator hat matrix.
    diagonal :
        Diagonal of the operator hat matrix.
    robust : ndarray
        The (n,) ndarray of robustness weights for robust fitting.
    divisor : ndarray
        The (p,) array of normalization divisors for numeric predictors.


    newdata : ndarray
        The (m,p) array of independent variables where the surface must be estimated.
    values : ndarray
        The (m,) ndarray of loess values evaluated at newdata
    stderr : ndarray
        The (m,) ndarray of the estimates of the standard error on the estimated
        values.
    residual_scale : float
        Estimate of the scale of the residuals
    df : integer
        Degrees of freedom of the t-distribution used to compute pointwise
        confidence intervals for the evaluated surface.
    nest : integer
        Number of new observations.
"""

loess_anova = _loess.anova
