# -*- Mode: Python -*-
# cython: embedsignature=True
import numpy as np
cimport numpy as np
from numpy cimport (ndarray, npy_intp,
                    NPY_DOUBLE, PyArray_SimpleNewFromData)
cimport c_loess

# NumPy must be initialized
np.import_array()


cdef floatarray_from_data(int rows, int cols, double *data):
    cdef ndarray a_ndr
    cdef npy_intp dims[2]
    cdef int nd
    dims = (rows, cols)
    nd = 2 if cols > 1 else 1
    a_ndr = <object>PyArray_SimpleNewFromData(nd, dims, NPY_DOUBLE, data)
    return a_ndr


cdef boolarray_from_data(int rows, int cols, int *data):
    cdef ndarray a_ndr
    cdef npy_intp dims[2]
    cdef int nd
    dims = (rows, cols)
    nd = 2 if cols > 1 else 1
    a_ndr = <object>PyArray_SimpleNewFromData(nd, dims, NPY_DOUBLE, data)
    return a_ndr.astype(np.bool)


cdef class loess_inputs:
    """
    Initialization class for loess data inputs

    Parameters
    ----------
    x : ndarray of shape (n, p)
        n independent observations for p no. of variables
    y : ndarray of shape (n,)
        A (n,) ndarray of response observations
    weights : ndarray of shape (n,) or None
        Weights to be given to individual observations
        in the sum of squared residuals that forms the local fitting
        criterion. If not None, the weights should be non negative. If
        the different observations have non-equal variances, the weights
        should be inversely proportional to the variances. By default,
        an unweighted fit is carried out (all the weights are one).
    """
    cdef c_loess.c_loess_inputs _base
    cdef readonly allocated

    def __cinit__(self, x, y, weights=None):
        cdef ndarray _x, _y, _w
        # When errors are raised the object gets destroyed and
        # __dealloc__ is called. It should not free up memory
        # if none was allocated prior to the error.
        self.allocated = False

        x = np.array(x, copy=False, subok=True,
                      dtype=np.float_, order='C')
        y = np.array(y, copy=False, subok=True,
                      dtype=np.float_, order='C')
        n = len(x)

        # Check the dimensions
        if x.ndim == 1:
            p = 1
        elif x.ndim == 2:
            p = x.shape[1]
        else:
            raise ValueError("The array of indepedent varibales "
                             "should be 2D at most!")

        if y.ndim != 1:
            raise ValueError("The array of dependent variables "
                             "should be 1D.")
        elif n != len(y):
            raise ValueError("The independent and depedent varibales "
                             "should have the same number of "
                             "observations.")

        if weights is None:
            weights = np.ones((n,), dtype=np.float_)

        if weights.ndim > 1 or weights.size != n:
            raise ValueError("Invalid size of the 'weights' vector!")

        weights = np.array(weights, copy=False, subok=True,
                           dtype=np.float_, order='C')

        # Python objects -> C structures -> *data in C structures
        _x = x.ravel()
        _y = y
        _w = weights

        x_dat = <double *>_x.data
        y_dat = <double *>_y.data
        w_dat = <double *>_w.data

        c_loess.loess_inputs_setup(x_dat, y_dat, w_dat, n, p, &self._base)
        self.allocated = True

    def __init__(self, x, y, weights=None):
        # For documentation
        pass

    def __dealloc__(self):
        if self.allocated:
            c_loess.loess_inputs_free(&self._base)

    @property
    def n(self):
        return self._base.n

    @property
    def p(self):
        return self._base.p

    @property
    def x(self):
        return floatarray_from_data(self.n, self.p, self._base.x)

    @property
    def y(self):
        return floatarray_from_data(self.n, 1, self._base.y)


cdef class loess_control:
    """
    Initialization class for loess control parameters

    Parameters
    ----------
    surface : str, optional
        One of ['interpolate', 'direct']
        Determines whether the fitted surface is computed directly
        at all points ('direct') or whether an interpolation method
        is used ('interpolate'). The default 'interpolate') is what
        most users should use unless special circumstances warrant.
    statistics : str, optional
        One of ['approximate', 'exact']
        Determines whether the statistical quantities are computed
        exactly ('exact') or approximately ('approximate'). 'exact'
        should only be used for testing the approximation in
        statistical development and is not meant for routine usage
        because computation time can be horrendous.
    trace_hat : str, optional
        One of ['wait.to.decide', 'exact', 'approximate']
        Determines how the trace of the hat matrix should be computed.
        The hat matrix is used in the computation of the statistical
        quantities. If 'exact', an exact computation is done; this
        could be slow when the number of observations n becomes large.
        If 'wait.to.decide' is selected, then a default is 'exact'
        for n < 500 and 'approximate' otherwise.
        This option is only useful when the fitted surface is
        interpolated. If surface is 'exact', an exact computation is
        always done for the trace. Setting trace_hat to 'approximate'
        for large dataset will substantially reduce the computation time.
    iterations : int, optional
        Number of iterations of the robust fitting method. If the family
        is 'gaussian', the number of iterations is set to 0.
    cell : float, optional
        Maximum cell size of the kd-tree. Suppose k = floor(n*cell*span),
        where n is the number of observations, and span the smoothing
        parameter. Then, a cell is further divided if the number of
        observations within it is greater than or equal to k. This
        option is only used if the surface is interpolated.
    """
    cdef c_loess.c_loess_control _base
    cdef bytes _surface
    cdef bytes _statistics
    cdef bytes _trace_hat

    def __cinit__(self, *args, **kwargs):
        # Does not allocate memory
        c_loess.loess_control_setup(&self._base)

    def __init__(self, surface='interpolate', statistics='approximate',
                 trace_hat='wait.to.decide', iterations=4, cell=0.2):
        self.surface = surface
        self.statistics = statistics
        self.trace_hat = trace_hat
        self.iterations = iterations
        self.cell = cell

    @property
    def surface(self):
        return self._base.surface.decode('utf-8')

    @surface.setter
    def surface(self, value):
        if value not in ('interpolate', 'direct'):
            raise ValueError(
                "Invalid value for the 'surface' argument: "
                "should be in ('interpolate', 'direct').")
        self._surface = value.encode('utf-8')
        self._base.surface = self._surface

    @property
    def statistics(self):
        return self._base.statistics.decode('utf-8')

    @statistics.setter
    def statistics(self, value):
        if value not in ('approximate', 'exact'):
            raise ValueError(
                "Invalid value for the 'statistics' argument: "
                "should be in ('approximate', 'exact').")
        self._statistics = value.encode('utf-8')
        self._base.statistics = self._statistics

    @property
    def trace_hat(self):
        return self._base.trace_hat.decode('utf-8')

    @trace_hat.setter
    def trace_hat(self, value):
        if value not in ('wait.to.decide', 'approximate', 'exact'):
            raise ValueError(
                "Invalid value for the 'trace_hat' argument: "
                "should be in ('approximate', 'exact').")
        self._trace_hat = value.encode('utf-8')
        self._base.trace_hat = self._trace_hat

    @property
    def iterations(self):
        return self._base.iterations

    @iterations.setter
    def iterations(self, value):
        if value < 0:
            raise ValueError(
                "Invalid number of iterations: "
                "should be positive")
        self._base.iterations = value

    @property
    def cell(self):
        return self._base.cell

    @cell.setter
    def cell(self, value):
        if value <= 0:
            raise ValueError(
                "Invalid value for the cell argument: "
                " should be positive")
        self._base.cell = value

    def __str__(self):
        strg = ["Control",
                "-------",
                "Surface type     : %s" % self.surface,
                "Statistics       : %s" % self.statistics,
                "Trace estimation : %s" % self.trace_hat,
                "Cell size        : %s" % self.cell,
                "Nb iterations    : %s" % self.iterations,]
        return '\n'.join(strg)


cdef class loess_kd_tree:
    cdef c_loess.c_loess_kd_tree _base

    def __cinit__(self, n, p):
        c_loess.loess_kd_tree_setup(n, p, &self._base)

    def __dealloc__(self):
        c_loess.loess_kd_tree_free(&self._base)


cdef class loess_model:
    """
    Initialization class for loess fitting parameters

    Parameters
    ----------
    p : int
        Number of variables
    family : str
        One of ('gaussian', 'symmetric')
        Determines the assumed distribution of the errors. If 'gaussian'
        the fit is performed with least-squares. If 'symmetric' is
        selected, the fit is performed robustly by redescending
        M-estimators.
    span : float
        Smoothing factor, as a fraction of the number of points to take
        into account. Should be in the range (0, 1]. Default is 0.75
    degree : int
        Overall degree of locally-fitted polynomial. 1 is locally-linear
        fitting and 2 is locally-quadratic fitting. Degree should be 2 at
        most. Default is 2.
    normalize : bool
        Determines whether the independent variables should be normalized.
        If True, the normalization is performed by setting the 10% trimmed
        standard deviation to one. If False, no normalization is carried
        out. This option is only useful for more than one variable. For
        spatial coordinates predictors or variables with a common scale,
        it should be set to False. Default is True.
    parametric : bool | list-of-bools of length p
        Indicates which independent variables should be
        conditionally-parametric (if there are two or more independent
        variables). If a sequence is given, the values should be ordered
        according to the predictor group in x.
    drop_square : bool | list-of-bools of length p
        Which squares to drop. When there are two or more independent
        variables and when a 2nd order polynomial(degree) is used,
        'drop_square' specifies those numeric predictors
        whose squares should be dropped from the set of fitting variables.
        If a sequence is given, the values should be ordered according to
        the predictor group in x.
    """

    cdef c_loess.c_loess_model _base
    cdef p
    cdef bytes _family

    def __cinit__(self, *args, **kwargs):
        # Does not allocate memory
        c_loess.loess_model_setup(&self._base)

    def __init__(self, p, family='gaussian', span=0.75,
                 degree=2, normalize=True, parametric=False,
                 drop_square=False):
        self.p = p
        self.family = family
        self.span = span
        self.degree = degree
        self.normalize = normalize
        self.parametric = parametric
        self.drop_square = drop_square

    @property
    def normalize(self):
        return bool(self._base.normalize)

    @normalize.setter
    def normalize(self, value):
        self._base.normalize = value

    @property
    def span(self):
        return self._base.span

    @span.setter
    def span(self, value):
        if value <= 0. or value > 1.:
            raise ValueError("Span should be between 0 and 1!")
        self._base.span = value

    @property
    def degree(self):
        return self._base.degree

    @degree.setter
    def degree(self, value):
        if value < 0 or value > 2:
            raise ValueError("Degree should be be 0, 1 or 2!")
        self._base.degree = value

    @property
    def family(self):
        return self._base.family.decode('utf-8')

    @family.setter
    def family(self, value):
        if value.lower()  not in ('symmetric', 'gaussian'):
            raise ValueError(
                "Invalid value for the 'family' argument: "
                "should be in ('symmetric', 'gaussian').")
        self._family = value.encode('utf-8')
        self._base.family = self._family

    @property
    def parametric(self):
        return boolarray_from_data(self.p, 1, self._base.parametric)

    @parametric.setter
    def parametric(self, value):
        cdef ndarray p_ndr

        if value in (True, False):
            value = [value] * self.p
        elif len(value) != self.p:
            raise ValueError(
                "'parametric' should be a boolean or a list "
                "of booleans with length equal to the number "
                "of independent variables")

        p_ndr = np.atleast_1d(np.array(value, copy=False, subok=True,
                                       dtype=np.bool))
        for i in range(self.p):
            self._base.parametric[i] = p_ndr[i]

    @property
    def drop_square(self):
        return boolarray_from_data(self.p, 1, self._base.drop_square)

    @drop_square.setter
    def drop_square(self, value):
        cdef ndarray d_ndr

        if value in (True, False):
            value = [value] * self.p
        elif len(value) != self.p:
            raise ValueError(
                "'drop_square' should be a boolean or a list "
                "of booleans with length equal to the number "
                "of independent variables")

        d_ndr = np.atleast_1d(np.array(value, copy=False,
                                       subok=True, dtype=np.bool))
        for i in range(self.p):
            self._base.drop_square[i] = d_ndr[i]

    def __repr__(self):
        return "<loess object: model parameters>"

    def __str__(self):
        strg = ["Model parameters",
                "----------------",
                "Family          : %s" % self.family,
                "Span            : %s" % self.span,
                "Degree          : %s" % self.degree,
                "Normalized      : %s" % self.normalize,
                "Parametric      : %s" % self.parametric[:self.p],
                "Drop_square     : %s" % self.drop_square[:self.p],
                ]
        return '\n'.join(strg)


cdef class loess_outputs:
    """
    Class of a loess fit outputs

    This object is automatically created with empty values when a
    new loess object is instantiated. The object gets filled when the
    loess.fit() method is called.

    Parameters
    ----------
    n : int
        Number of independent observation
    p : int
        Number of variables

    Attributes
    ----------
    fitted_values : ndarray of shape (n,)
        Fitted values.
    fitted_residuals : ndarray of shape (n,)
        Fitted residuals (observations - fitted values).
    enp : float
        Equivalent number of parameters.
    residual_scale : float
        Estimate of the scale of residuals.
    one_delta: float
        Statistical parameter used in the computation of standard errors.
    two_delta : float
        Statistical parameter used in the computation of standard errors.
    pseudovalues : ndarray of shape (n,)
        Adjusted values of the response when robust estimation is used.
    trace_hat : float
        Trace of the operator hat matrix.
    diagonal : ndarray of shape (n,)
        Diagonal of the operator hat matrix.
    robust : ndarray of shape (n,)
        Robustness weights for robust fitting.
    divisor : ndarray of shape (p,)
        Normalization divisors for numeric predictors.
    """
    cdef c_loess.c_loess_outputs _base
    # private
    cdef readonly family
    cdef readonly n, p
    cdef readonly activated

    def __cinit__(self, n, p, family):
        c_loess.loess_outputs_setup(n, p, &self._base)

    def __init__(self, n, p, family):
        self.n = n
        self.p = p
        self.family = family
        self.activated = False

    def __dealloc__(self):
        c_loess.loess_outputs_free(&self._base)

    @property
    def fitted_values(self):
        return floatarray_from_data(self.n, 1, self._base.fitted_values)

    @property
    def fitted_residuals(self):
        return floatarray_from_data(self.n, 1, self._base.fitted_residuals)

    @property
    def pseudovalues(self):
        if self.family != 'symmetric':
            raise ValueError(
                "pseudovalues are available only when "
                "robust fitting. Use family='symmetric' "
                "for robust fitting")
        return floatarray_from_data(self.n, 1, self._base.pseudovalues)

    @property
    def diagonal(self):
        return floatarray_from_data(self.n, 1, self._base.diagonal)

    @property
    def robust(self):
        return floatarray_from_data(self.n, 1, self._base.robust)

    @property
    def divisor(self):
        return floatarray_from_data(self.n, 1, self._base.divisor)

    @property
    def enp(self):
        return self._base.enp

    @property
    def residual_scale(self):
        return self._base.residual_scale

    @property
    def one_delta(self):
        return self._base.one_delta

    @property
    def two_delta(self):
        return self._base.two_delta

    @property
    def trace_hat(self):
        return self._base.trace_hat

    def __str__(self):
        strg = ["Outputs",
                "-------",
                "Fitted values         : %s\n" % self.fitted_values,
                "Fitted residuals      : %s\n" % self.fitted_residuals,
                "Eqv. nb of parameters : %s" % self.enp,
                "Residual error        : %s" % self.s,
                "Deltas                : %s - %s" % (self.one_delta,
                                                     self.two_delta),
                "Normalization factors : %s" % self.divisor,]
        return '\n'.join(strg)


cdef class loess_confidence_intervals:
    """
    Pointwise confidence intervals of a loess-predicted object

    Parameters
    ----------
    pred : loess_prediction
        Prediction object
    alpha : float
        The alpha level for the confidence interval.
        It must be in the range (0, 1)

    Attributes
    ----------
    fit : ndarray
        Predicted values.
    lower : ndarray
        Lower bounds of the confidence intervals.
    upper : ndarray
        Upper bounds of the confidence intervals.
    """
    cdef c_loess.c_confidence_intervals _base
    cdef readonly m

    def __cinit__(loess_confidence_intervals self, loess_prediction pred,
                  float alpha):
        coverage = 1 - alpha
        if coverage < .5:
            coverage = 1 - coverage

        if not 0 < alpha < 1. :
            raise ValueError("The alpha value should be "
                             "between 0 and 1.")
        if not pred._base.se:
            raise ValueError("Cannot compute confidence intervals "
                             "without standard errors.")

        c_loess.c_pointwise(&pred._base, coverage, &self._base)
        self.m = pred.m

    def __init__(self, pred, alpha):
        # For documentation
        pass

    def __dealloc__(self):
        c_loess.pw_free_mem(&self._base)

    @property
    def fit(self):
        return  floatarray_from_data(self.m, 1, self._base.fit)

    @property
    def upper(self):
        return  floatarray_from_data(self.m, 1, self._base.upper)

    @property
    def lower(self):
        return  floatarray_from_data(self.m, 1, self._base.lower)


cdef class loess_prediction:
    """
    Class for loess prediction results

    Holds the predicted values and standard errors of a loess object

    Parameters
    ----------
    newdata : ndarray of shape (m, p)
        Independent variables where the surface must be estimated,
        with m the number of new data points, and p the number of
        independent variables.
    loess : loess.loess
        Loess object that has been successfully fitted,
        i.e `loess.fit` has been called and it returned without
        any errors.
    stderror : boolean
        Whether the standard error should be computed

    Attributes
    ----------
    values : ndarray of shape (m,)
        loess values evaluated at newdata
    stderr : ndarray of shape (m,)
        Estimates of the standard error on the estimated values.
        `ValueError` is raise when the standard error was not computed.
    residual_scale : float
        Estimate of the scale of the residuals
    df : integer
        Degrees of freedom of the t-distribution used to compute pointwise
        confidence intervals for the evaluated surface.
    """

    cdef c_loess.c_prediction _base
    cdef readonly allocated

    def __cinit__(self, newdata, loess loess, stderror=False):
        cdef ndarray p_ndr
        cdef double *p_dat

        self.allocated = False

        # Note : we need a copy as we may have to normalize
        p_ndr = np.array(newdata, copy=True, subok=True, order='C')
        p_ndr = p_ndr.astype(float)

        # Dimensions should match those of the input
        if p_ndr.size == 0 or p_ndr.ndim == 0:
            raise ValueError("Can't predict without input data !")

        if p_ndr.ndim > 2:
            raise ValueError("New data has more than 2 dimensions.")

        _p = 1 if p_ndr.ndim == 1 else p_ndr.shape[1]
        if _p != loess.inputs.p:
            msg = ("Incompatible data size: there should be as many "
                   "columns as parameters. Got %d instead of %d "
                   "parameters" % (_p, loess.inputs.p))
            raise ValueError(msg)

        m = len(p_ndr)
        p_ndr = p_ndr.ravel()
        p_dat = <double *>p_ndr.data

        self._base.se = 1 if stderror else 0
        self._base.m = m
        c_loess.c_predict(p_dat, &loess._base, &self._base)
        self.allocated = True

        if loess._base.status.err_status:
            raise ValueError(loess._base.status.err_msg)

    def __init__(self, newdata, loess, stderror=False):
        # For documentation
        pass

    def __dealloc__(self):
        if self.allocated:
            c_loess.pred_free_mem(&self._base)

    @property
    def values(self):
        return floatarray_from_data(self.m, 1, self._base.fit)

    @property
    def stderr(self):
        if not self._base.se:
            raise ValueError("Standard error was not computed."
                             "Use 'stderror=True' when predicting.")
        return floatarray_from_data(self.m, 1, self._base.se_fit)

    @property
    def residual_scale(self):
        return self._base.residual_scale

    @property
    def df(self):
        return self._base.df

    @property
    def m(self):
        return self._base.m

    def confidence(self, alpha=0.05):
        """
        Returns the pointwise confidence intervals

        Parameters
        ----------
        alpha : float
            The alpha level for the confidence interval. The
            default ``alpha=0.05`` returns a 95% confidence
            interval. Therefore it must be in the range (0, 1).

        Returns
        -------
        out : loess_confidence_intervals
            Confidence intervals object. It has attributes `fit`,
            `lower` and `upper`
        """
        return loess_confidence_intervals(self, alpha)

    def __str__(self):
        try:
            stderr = "Predicted std error   : %s\n" % self.stderr
        except ValueError:
            stderr = ""

        strg = ["Outputs",
                "-------",
                "Predicted values      : %s\n" % self.values,
                stderr,
                "Residual scale        : %s" % self.residual_scale,
                "Degrees of freedom    : %s" % self.df,
                ]
        return '\n'.join(strg)


cdef class loess:
    """
    Locally-weighted regression

    A loess object is initialized with the combined parameters of
    :class:`loess_inputs`, :class:`loess_model` and
    :class:`loess_control`. The parameters of :class:`loess_inputs`
    i.e ``x``, ``y`` and ``weights`` can be positional in that order.
    In the descriptions below, `n` is the number of observations,
    and `p` is the number of predictor variables.

    Parameters
    ----------
    x : ndarray of shape (n, p)
        n independent observations for p no. of variables
    y : ndarray of shape (n,)
        A (n,) ndarray of response observations
    weights : ndarray of shape (n,) or None
        Weights to be given to individual observations
        in the sum of squared residuals that forms the local fitting
        criterion. If not None, the weights should be non negative. If
        the different observations have non-equal variances, the weights
        should be inversely proportional to the variances. By default,
        an unweighted fit is carried out (all the weights are one).
    **options : dict
        The parameters of :class:`loess_model` and
        :class: `loess_control`.

    Attributes
    ----------
    fitted_values : ndarray of shape (n,)
        Fitted values.
    fitted_residuals : ndarray of shape (n,)
        Fitted residuals (observations - fitted values).
    enp : float
        Equivalent number of parameters.
    s : float
        Estimate of the scale of residuals.
    one_delta: float
        Statistical parameter used in the computation of standard errors.
    two_delta : float
        Statistical parameter used in the computation of standard errors.
    pseudovalues : ndarray of shape (n,)
        Adjusted values of the response when robust estimation is used.
    trace_hat : float
        Trace of the operator hat matrix.
    diagonal :
        Diagonal of the operator hat matrix.
    robust : ndarray of shape (n,)
        Robustness weights for robust fitting.
    divisor : ndarray of shape(p,)
        Normalization divisors for numeric predictors.
    """
    cdef c_loess.c_loess _base
    cdef readonly loess_inputs inputs
    cdef readonly loess_model model
    cdef readonly loess_control control
    cdef readonly loess_kd_tree kd_tree
    cdef readonly loess_outputs outputs

    def __init__(self, object x, object y, object weights=None, **options):
        # Process options
        model_options = {}
        control_options= {}
        for (k, v) in options.items():
            if k in ('family', 'span', 'degree', 'normalize',
                     'parametric', 'drop_square',):
                model_options[k] = v
            elif k in ('surface', 'statistics', 'trace_hat',
                       'iterations', 'cell'):
                control_options[k] = v


        # Initialize the inputs
        self.inputs = loess_inputs(x, y, weights)
        self._base.inputs = &self.inputs._base

        n = self.inputs.n
        p = self.inputs.p

        # Initialize the control parameters
        self.control = loess_control(**control_options)
        self._base.control = &self.control._base

        # Initialize the model parameters
        self.model = loess_model(p, **model_options)
        self._base.model = &self.model._base

        # Initialize the outputs
        self.outputs = loess_outputs(n, p, self.model.family)
        self._base.outputs = &self.outputs._base

        # Initialize the kd tree
        self.kd_tree = loess_kd_tree(n, p)
        self._base.kd_tree = &self.kd_tree._base

    def fit(self):
        """
        Computes the loess parameters on the current inputs and
        sets of parameters.
        """
        c_loess.loess_fit(&self._base)
        self.outputs.activated = True
        if self._base.status.err_status:
            raise ValueError(self._base.status.err_msg)
        return

    def input_summary(self):
        """
        Returns some generic information about the loess parameters.
        """
        toprint = [str(self.model), str(self.control)]
        return "\n\n".join(toprint)

    def output_summary(self):
        """Returns some generic information about the loess fit."""
        fit_flag = bool(self.outputs.activated)

        if self.model.family == "gaussian":
            rse = ("Residual Standard Error        : %.4f" %
                   self.outputs.residual_scale)
        else:
            rse = ("Residual Scale Estimate        : %.4f" %
                   self.outputs.residual_scale)

        strg = ["Output Summary",
                "--------------",
                "Number of Observations         : %d" % self.inputs.n,
                "Fit flag                       : %d" % fit_flag,
                "Equivalent Number of Parameters: %.1f" % self.outputs.enp,
                rse
                ]
        return '\n'.join(strg)

    def predict(self, newdata, stderror=False):
        """
        Compute loess estimates at the given new data points newdata.

        Parameters
        ----------
        newdata : ndarray of shape (m, p)
            Independent variables where the surface must be estimated,
            with m the number of new data points, and p the number of
            independent variables.
        stderror : boolean
            Whether the standard error should be computed

        Returns
        -------
        A `loess_prediction` object.
        """
        # Make sure there's been a fit earlier
        if self.outputs.activated == 0:
            c_loess.loess_fit(&self._base)
            self.outputs.activated = True
            if self._base.status.err_status:
                raise ValueError(self._base.status.err_msg)

        return loess_prediction(newdata, self, stderror)


cdef class loess_anova:
    """
    Analysis of variance for two loess objects

    Parameters
    ----------
    loess_one : loess.loess
        First loess object
    loess_two : loess.loess
        Second loess object

    Attributes
    ----------
    F_value : float
        Value of the F-statistic
    Pr_F : float
        Probability of getting a value as large as the
        `F_value`. The is the p-value of the F-statistic.
    """
    cdef readonly double dfn, dfd, F_value, Pr_F

    def __init__(self, loess_one, loess_two):
        cdef double one_d1, one_d2, one_s, two_d1, two_d2, two_s
        cdef double rssdiff, d1diff, tmp, df1, df2

        if (not isinstance(loess_one, loess) or
                not isinstance(loess_two, loess)):
            raise ValueError("Arguments should be valid loess objects!"
                             "got '%s' instead" % type(loess_one))

        out_one = loess_one.outputs
        out_two = loess_two.outputs

        one_d1 = out_one.one_delta
        one_d2 = out_one.two_delta
        one_s = out_one.residual_scale

        two_d1 = out_two.one_delta
        two_d2 = out_two.two_delta
        two_s = out_two.residual_scale

        rssdiff = abs(one_s * one_s * one_d1 - two_s * two_s * two_d1)
        d1diff = abs(one_d1 - two_d1)
        self.dfn = d1diff * d1diff / abs(one_d2 - two_d2)
        df1 = self.dfn

        if out_one.enp > out_two.enp:
            self.dfd = one_d1 * one_d1 / one_d2
            tmp = one_s
        else:
            self.dfd = two_d1 * two_d1 / two_d2
            tmp = two_s
        df2 = self.dfd
        self.F_value = (rssdiff / d1diff) / (tmp * tmp)
        self.Pr_F = 1. - c_loess.pf(self.F_value, df1, df2)
