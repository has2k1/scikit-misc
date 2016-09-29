# -*- Mode: Python -*-  
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


"""
        

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


#####---------------------------------------------------------------------------
#---- ---- loess model ---
#####---------------------------------------------------------------------------
cdef class loess_inputs:
    """
    Loess inputs

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
    cdef ndarray w_ndr
    cdef readonly int allocated

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

    def __dealloc__(self):
        if self.allocated:
            c_loess.loess_inputs_free(&self._base)

    property n:
        def __get__(self):
            return self._base.n

    property p:
        def __get__(self):
            return self._base.p

    property x:
        def __get__(self):
            return floatarray_from_data(self.n, self.p, self._base.x)

    property y:
        def __get__(self):
            return floatarray_from_data(self.n, 1, self._base.y)


cdef class loess_control:
    """Loess control parameters.
    
:IVariables:
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
    
    """

    cdef c_loess.c_loess_control _base

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

    property surface:
        """
    surface : string ["interpolate"]
        Determines whether the fitted surface is computed directly at all points
        ("direct") or whether an interpolation method is used ("interpolate").
        The default ("interpolate") is what most users should use unless special 
        circumstances warrant.
        """
        def __get__(self):
            return self._base.surface
        def __set__(self, surface):
            if surface not in ('interpolate', 'direct'):
                raise ValueError(
                    "Invalid value for the 'surface' argument: "
                    "should be in ('interpolate', 'direct').")
            self._base.surface = surface

    property statistics:
        """
    statistics : string ["approximate"]
        Determines whether the statistical quantities are computed exactly 
        ("exact") or approximately ("approximate"). "exact" should only be used 
        for testing the approximation in statistical development and is not meant 
        for routine usage because computation time can be horrendous.
        """
        def __get__(self):
            return self._base.statistics
        def __set__(self, statistics):
            if statistics not in ('approximate', 'exact'):
                raise ValueError(
                    "Invalid value for the 'statistics' argument: "
                    "should be in ('approximate', 'exact').")
            self._base.statistics = statistics

    property trace_hat:
        """
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
        """
        def __get__(self):
            return self._base.trace_hat
        def __set__(self, trace_hat):
            if trace_hat not in ('wait.to.decide', 'approximate', 'exact'):
                raise ValueError(
                    "Invalid value for the 'trace_hat' argument: "
                    "should be in ('approximate', 'exact').")
            self._base.trace_hat = trace_hat

    property iterations:
        """
    iterations : integer
        Number of iterations of the robust fitting method. If the family is 
        "gaussian", the number of iterations is set to 0.
        """
        def __get__(self):
            return self._base.iterations
        def __set__(self, iterations):
            if iterations < 0:
                raise ValueError("Invalid number of iterations: should be positive")
            self._base.iterations = iterations
    #.........
    property cell:
        """
    cell : integer
        Maximum cell size of the kd-tree. Suppose k = floor(n*cell*span),
        where n is the number of observations, and span the smoothing parameter.
        Then, a cell is further divided if the number of observations within it 
        is greater than or equal to k. This option is only used if the surface 
        is interpolated.
        """     
        def __get__(self):
            return self._base.cell
        def __set__(self, cell):
            if cell <= 0:
                raise ValueError("Invalid value for the cell argument: should be positive")
            self._base.cell = cell

    def __str__(self):
        strg = ["Control",
                "-------",
                "Surface type     : %s" % self.surface,
                "Statistics       : %s" % self.statistics,
                "Trace estimation : %s" % self.trace_hat,
                "Cell size        : %s" % self.cell,
                "Nb iterations    : %s" % self.iterations,]
        return '\n'.join(strg)
        
#    
######---------------------------------------------------------------------------
##---- ---- loess kd_tree ---
######---------------------------------------------------------------------------
cdef class loess_kd_tree:
    cdef c_loess.c_loess_kd_tree _base

    def __cinit__(self, n, p):
        c_loess.loess_kd_tree_setup(n, p, &self._base)

    def __dealloc__(self):
        c_loess.loess_kd_tree_free(&self._base)


cdef class loess_model:
    """loess_model contains parameters required for a loess fit.
    
:IVariables:
    normalize : boolean [True]
        Determines whether the independent variables should be normalized.  
        If True, the normalization is performed by setting the 10% trimmed 
        standard deviation to one. If False, no normalization is carried out. 
        This option is only useful for more than one variable. For spatial
        coordinates predictors or variables with a common scale, it should be 
        set to False.
    span : float [0.75]
        Smoothing factor, as a fraction of the number of points to take into
        account. 
    degree : integer [2]
        Overall degree of locally-fitted polynomial. 1 is locally-linear 
        fitting and 2 is locally-quadratic fitting.  Degree should be 2 at most.
    family : string ["gaussian"]
        Determines the assumed distribution of the errors. The values are 
        "gaussian" or "symmetric". If "gaussian" is selected, the fit is 
        performed with least-squares. If "symmetric" is selected, the fit
        is performed robustly by redescending M-estimators.
    parametric_flags : sequence [ [False]*p ]
        Indicates which independent variables should be conditionally-parametric
        (if there are two or more independent variables). The argument should be
        a sequence of booleans, with the same size as the number of independent 
        variables, specified in the order of the predictor group ordered in x. 
        Note: elements of the sequence cannot be modified individually: the whole
        sequence must be given.
    drop_square : sequence [ [False]* p]
        When there are two or more independent variables and when a 2nd order
        polynomial is used, "drop_square_flags" specifies those numeric predictors 
        whose squares should be dropped from the set of fitting variables. 
        The method of specification is the same as for parametric.  
        Note: elements of the sequence cannot be modified individually: the whole
        sequence must be given.

    """

    cdef c_loess.c_loess_model _base
    cdef long p

    def __cinit__(self, *args, **kwargs):
        # Does not allocate memory
        c_loess.loess_model_setup(&self._base)

    def __init__(self, p, family="gaussian", span=0.75,
                 degree=2, normalize=True, parametric=False,
                 drop_square=False):
        self.p = p

        self.family = family
        self.span = span
        self.degree = degree
        self.normalize = normalize
        self.parametric = parametric
        self.drop_square = drop_square

    #.........
    property normalize:
        def __get__(self):
            return bool(self._base.normalize)
        def __set__(self, normalize):
            self._base.normalize = normalize
    #.........
    property span:
        def __get__(self):
            return self._base.span
        def __set__(self, span):
            if span <= 0. or span > 1.:
                raise ValueError("Span should be between 0 and 1!")
            self._base.span = span
    #.........
    property degree:
        def __get__(self):
            return self._base.degree
        def __set__(self, degree):
            if degree < 0 or degree > 2:
                raise ValueError("Degree should be between 0 and 2!")
            self._base.degree = degree
    #.........
    property family:
        def __get__(self):
            return self._base.family
        def __set__(self, family):
            if family.lower() not in ('symmetric', 'gaussian'):
                raise ValueError("Invalid value for the 'family' argument: "\
                                 "should be in ('symmetric', 'gaussian').")
            self._base.family = family
    #.........
    property parametric:
        def __get__(self):
            return boolarray_from_data(self.p, 1, self._base.parametric)
        def __set__(self, paramf):
            cdef ndarray p_ndr
            cdef int i

            if paramf in (True, False):
                paramf = [paramf] * self.p
            elif len(paramf) != self.p:
                raise ValueError(
                    "'parametric' should be a boolean or a list "
                    "of booleans with length equal to the number "
                    "of independent variables")

            p_ndr = np.atleast_1d(np.array(paramf, copy=False, subok=True,
                                           dtype=np.bool))
            for i from 0 <= i < self.p:
                self._base.parametric[i] = p_ndr[i]
    #.........
    property drop_square:
        def __get__(self):
            return boolarray_from_data(self.p, 1, self._base.drop_square)
        def __set__(self, drop_sq):
            cdef ndarray d_ndr
            cdef int i

            if drop_sq in (True, False):
                drop_sq = [drop_sq] * self.p
            elif len(drop_sq) != self.p:
                raise ValueError(
                    "'drop_square' should be a boolean or a list "
                    "of booleans with length equal to the number "
                    "of independent variables")

            d_ndr = np.atleast_1d(np.array(drop_sq, copy=False,
                                           subok=True, dtype=np.bool))
            for i from 0 <= i < self.p:
                self._base.drop_square[i] = d_ndr[i]

    #.........
    def __repr__(self):
        return "<loess object: model parameters>"
    #.........
    def __str__(self):
        strg = ["Model parameters",
                "----------------",
                "Family          : %s" % self.family,
                "Span            : %s" % self.span,
                "Degree          : %s" % self.degree,
                "Normalized      : %s" % self.normalize,
                "Parametric      : %s" % self.parametric_flags[:self.p],
                "Drop_square     : %s" % self.drop_square_flags[:self.p]
                ]
        return '\n'.join(strg)
        
#####---------------------------------------------------------------------------
#---- ---- loess outputs ---
#####---------------------------------------------------------------------------
cdef class loess_outputs:
    """Outputs of a loess fit. This object is automatically created with empty
values when a new loess object is instantiated. The object gets filled when the 
loess.fit() method is called.
    
:IVariables:
    fitted_values : ndarray
        The (n,) ndarray of fitted values.
    fitted_residuals : ndarray
        The (n,) ndarray of fitted residuals (observations - fitted values).
    enp : float
        Equivalent number of parameters.
    residual_scale : float
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
    diagonal : ndarray
        Diagonal of the operator hat matrix.
    robust : ndarray
        The (n,) ndarray of robustness weights for robust fitting.
    divisor : ndarray
        The (p,) array of normalization divisors for numeric predictors.
    """
    cdef c_loess.c_loess_outputs _base
    cdef readonly char *family
    cdef readonly long n, p
    cdef readonly int activated

    def __cinit__(self, n, p, family):
        c_loess.loess_outputs_setup(n, p, &self._base)

    def __init__(self, n, p, family):
        self.n = n
        self.p = p
        self.family = family
        self.activated = False

    def __dealloc__(self):
        c_loess.loess_outputs_free(&self._base)

    property fitted_values:
        def __get__(self):
            return floatarray_from_data(self.n, 1, self._base.fitted_values)
    #.........
    property fitted_residuals:
        def __get__(self):
            return floatarray_from_data(self.n, 1, self._base.fitted_residuals)
    #.........
    property pseudovalues:
        def __get__(self):
            if self.family not in ('symmetric'):
                raise ValueError(
                    "pseudovalues are available only when "
                    "robust fitting. Use family='symmetric' "
                    "for robust fitting")
            return floatarray_from_data(self.n, 1, self._base.pseudovalues)
    #.........
    property diagonal:
        def __get__(self):
            return floatarray_from_data(self.n, 1, self._base.diagonal)
    #.........
    property robust:
        def __get__(self):
            return floatarray_from_data(self.n, 1, self._base.robust)
    #.........
    property divisor:
        def __get__(self):
            return floatarray_from_data(self.n, 1, self._base.divisor)
    #.........
    property enp:
        def __get__(self):
            return self._base.enp
    #.........
    property residual_scale:
        def __get__(self):
            return self._base.residual_scale
    #.........
    property one_delta:
        def __get__(self):
            return self._base.one_delta 
    #.........
    property two_delta:
        def __get__(self):
            return self._base.two_delta
    #.........
    property trace_hat:
        def __get__(self):
            return self._base.trace_hat
    #.........
    def __str__(self):
        strg = ["Outputs",
                "-------",
                "Fitted values         : %s\n" % self.fitted_values,
                "Fitted residuals      : %s\n" % self.fitted_residuals,
                "Eqv. nb of parameters : %s" % self.enp,
                "Residual error        : %s" % self.s,
                "Deltas                : %s - %s" % (self.one_delta, self.two_delta),
                "Normalization factors : %s" % self.divisor,]
        return '\n'.join(strg)


        
#####---------------------------------------------------------------------------
#---- ---- loess confidence ---
#####---------------------------------------------------------------------------
cdef class conf_intervals:
    """Pointwise confidence intervals of a loess-predicted object:
    
:IVariables:
    fit : ndarray
        Predicted values.
    lower : ndarray
        Lower bounds of the confidence intervals.
    upper : ndarray
        Upper bounds of the confidence intervals.
    """
    cdef c_loess.c_conf_inv _base
    cdef readonly ndarray lower, fit, upper
    #.........
    def __dealloc__(self):
        c_loess.pw_free_mem(&self._base)
    #.........
    cdef setup(self, c_loess.c_conf_inv base, long nest):
        self._base = base
        self.fit = floatarray_from_data(nest, 1, base.fit)
        self.upper = floatarray_from_data(nest, 1, base.upper)
        self.lower = floatarray_from_data(nest, 1, base.lower)
    #.........
    def __str__(self):
        cdef ndarray tmp_ndr
        tmp_ndr = np.r_[[self.lower,self.fit,self.upper]].T
        return "Confidence intervals....\nLower b./ fit / upper b.\n%s" % \
               tmp_ndr 

#####---------------------------------------------------------------------------
#---- ---- loess predictions ---
#####---------------------------------------------------------------------------
cdef class loess_predicted:
    """Predicted values and standard errors of a loess object

:IVariables:
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
        
    cdef c_loess.c_prediction _base
    cdef readonly conf_intervals confidence_intervals
    cdef readonly int allocated

    def __cinit__(self, newdata, loess loess, stderror=False):
        cdef ndarray p_ndr
        cdef double *p_dat

        self.allocated = False

        # Note : we need a copy as we may have to normalize
        p_ndr = np.array(newdata, copy=True, subok=True, order='C')

        # Note : we need a copy as we may have to normalize
        p_ndr = np.array(newdata, copy=True,
                         subok=True, order='C').ravel()
        p_dat = <double *>p_ndr.data

        # Test the compatibility of sizes
        if p_ndr.size == 0:
            raise ValueError("Can't predict without input data !")

        (m, notOK) = divmod(len(p_ndr), loess.inputs.p)

        if notOK:
            raise ValueError("Incompatible data size: there should "
                             "be as many rows as parameters")

        self._base.se = 1 if stderror else 0
        self._base.m = m
        c_loess.c_predict(p_dat, &loess._base, &self._base)
        self.allocated = True

        if loess._base.status.err_status:
            raise ValueError(loess._base.status.err_msg)

    def __dealloc__(self):
        if self.allocated:
            c_loess.pred_free_mem(&self._base)

    property values:
        def __get__(self):
            return floatarray_from_data(self.m, 1, self._base.fit)

    property stderr:
        def __get__(self):
            if not self._base.se:
                raise ValueError("Standard error was not computed."
                                 "Use 'stderror=True' when predicting.")
            return floatarray_from_data(self.m, 1, self._base.se_fit)

    property residual_scale:
        def __get__(self):
            return self._base.residual_scale
    #.........
    property df:
        def __get__(self):
            return self._base.df        

    property m:
        def __get__(self):
            return self._base.m
    #.........
    def confidence(self, coverage=0.95):
        """Returns the pointwise confidence intervals for each predicted values,
at the given confidence interval coverage.
        
:Parameters:
    coverage : float
        Confidence level of the confidence intervals limits, as a fraction.
        
:Returns:
    A new conf_intervals object, consisting of:
    fit : ndarray
        Predicted values.
    lower : ndarray
        Lower bounds of the confidence intervals.
    upper : ndarray
        Upper bounds of the confidence intervals.
        """
        cdef c_loess.c_conf_inv _confintv
        if coverage < 0.5:
            coverage = 1. - coverage 
        if coverage > 1. :
            raise ValueError("The coverage precentage should be between 0 and 1!")
        if not self._base.se:
            raise ValueError("Cannot compute confidence intervals "
                             "without standard errors.")
        c_loess.c_pointwise(&self._base, self.nest, coverage, &_confintv)
        self.confidence_intervals = conf_intervals()
        self.confidence_intervals.setup(_confintv, self.nest)
        return self.confidence_intervals
    #.........
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
    

#####---------------------------------------------------------------------------
#---- ---- loess base class ---
#####---------------------------------------------------------------------------
cdef class loess:
    """
    
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

    """
    cdef c_loess.c_loess _base
    cdef readonly loess_inputs inputs
    cdef readonly loess_model model
    cdef readonly loess_control control
    cdef readonly loess_kd_tree kd_tree
    cdef readonly loess_outputs outputs
    cdef readonly loess_predicted predicted
    
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
    #......................................................
    def fit(self):
        """Computes the loess parameters on the current inputs and sets of parameters."""
        c_loess.loess_fit(&self._base)
        self.outputs.activated = True
        if self._base.status.err_status:
            raise ValueError(self._base.status.err_msg)
        return
    #......................................................
    def input_summary(self):
        """Returns some generic information about the loess parameters.
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
    #......................................................
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

        self.predicted = loess_prediction(newdata, self, stderror)
        return self.predicted


cdef class anova:
    cdef readonly double dfn, dfd, F_value, Pr_F
    #
    def __init__(self, loess_one, loess_two):
        cdef double one_d1, one_d2, one_s, two_d1, two_d2, two_s, rssdiff,\
                    d1diff, tmp, df1, df2
        #
        if not isinstance(loess_one, loess) or not isinstance(loess_two, loess):
            raise ValueError("Arguments should be valid loess objects!"\
                             "got '%s' instead" % type(loess_one))
        #
        out_one = loess_one.outputs
        out_two = loess_two.outputs
        #
        one_d1 = out_one.one_delta
        one_d2 = out_one.two_delta
        one_s = out_one.residual_scale
        #
        two_d1 = out_two.one_delta
        two_d2 = out_two.two_delta
        two_s = out_two.residual_scale
        #
        rssdiff = abs(one_s * one_s * one_d1 - two_s * two_s * two_d1)
        d1diff = abs(one_d1 - two_d1)
        self.dfn = d1diff * d1diff / abs(one_d2 - two_d2)
        df1 = self.dfn
        #
        if out_one.enp > out_two.enp:
            self.dfd = one_d1 * one_d1 / one_d2
            tmp = one_s
        else:
            self.dfd = two_d1 * two_d1 / two_d2
            tmp = two_s
        df2 = self.dfd
        F_value = (rssdiff / d1diff) / (tmp * tmp)
        
        self.Pr_F = 1. - c_loess.ibeta(F_value*df1/(df2+F_value*df1), df1/2, df2/2)
        self.F_value = F_value

        
