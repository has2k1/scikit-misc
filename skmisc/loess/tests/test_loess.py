import os

import pytest
import numpy as np
import numpy.testing as npt

from skmisc.loess import loess, loess_anova

data_path = os.path.dirname(os.path.abspath(__file__))


def madeup_data():
    dfile = os.path.join(data_path, 'madeup_data')
    rfile = os.path.join(data_path, 'madeup_result')

    with open(dfile, 'r') as f:
        f.readline()
        x = np.fromiter(
            (float(v) for v in f.readline().rstrip().split()),
            np.float_).reshape(-1, 2)
        f.readline()
        y = np.fromiter(
            (float(v) for v in f.readline().rstrip().split()),
            np.float_)

    results = []
    with open(rfile, 'r') as f:
        for i in range(8):
            f.readline()
            z = np.fromiter(
                (float(v) for v in f.readline().rstrip().split()),
                np.float_)
            results.append(z)

    newdata1 = np.array([[-2.5, 0.], [2.5, 0.], [0., 0.]])
    newdata2 = np.array([[-0.5, 0.5], [0., 0.]])
    return (x, y, results, newdata1, newdata2)


def gas_data():
    NOx = np.array([4.818, 2.849, 3.275, 4.691, 4.255, 5.064, 2.118, 4.602,
                    2.286, 0.970, 3.965, 5.344, 3.834, 1.990, 5.199, 5.283,
                    3.752, 0.537, 1.640, 5.055, 4.937, 1.561])
    E = np.array([0.831, 1.045, 1.021, 0.970, 0.825, 0.891, 0.71, 0.801,
                  1.074, 1.148, 1.000, 0.928, 0.767, 0.701, 0.807, 0.902,
                  0.997, 1.224, 1.089, 0.973, 0.980, 0.665])
    gas_fit_E = np.array([0.665, 0.949, 1.224])
    newdata = np.array([0.6650000, 0.7581667, 0.8513333, 0.9445000,
                        1.0376667, 1.1308333, 1.2240000])
    alpha = 0.01

    rfile = os.path.join(data_path, 'gas_result')
    results = []
    with open(rfile, 'r') as f:
        for i in range(8):
            f.readline()
            z = np.fromiter(
                (float(v) for v in f.readline().rstrip().split()),
                np.float_)
            results.append(z)
    return (E, NOx, gas_fit_E, newdata, alpha, results)


class TestLoess2d(object):
    "Test class for lowess."
    d = madeup_data()

    def test_2dbasic(self):
        "2D standard"
        (x, y, results, _, _) = self.d
        madeup = loess(x, y)
        madeup.model.span = 0.5
        madeup.model.normalize = True
        madeup.fit()
        npt.assert_almost_equal(madeup.outputs.fitted_values, results[0], 5)
        npt.assert_almost_equal(madeup.outputs.enp, 14.9, 1)
        npt.assert_almost_equal(madeup.outputs.residual_scale, 0.9693, 4)

    def test_2d_modflags(self):
        "2D - modification of model flags"
        (x, y, results, _, _) = self.d
        madeup = loess(x, y)
        madeup.model.span = 0.8
        madeup.model.drop_square = [True, False]
        madeup.model.parametric = [True, False]
        npt.assert_equal(madeup.model.parametric[:2], [1, 0])
        madeup.fit()
        npt.assert_almost_equal(madeup.outputs.fitted_values, results[1], 5)
        npt.assert_almost_equal(madeup.outputs.enp, 6.9, 1)
        npt.assert_almost_equal(madeup.outputs.residual_scale, 1.4804, 4)

    def test_2d_modfamily(self):
        "2D - family modification"
        (x, y, results, _, _) = self.d
        madeup = loess(x, y)
        madeup.model.span = 0.8
        madeup.model.drop_square = [True, False]
        madeup.model.parametric = [True, False]
        madeup.model.family = "symmetric"
        madeup.fit()
        npt.assert_almost_equal(madeup.outputs.fitted_values, results[2], 5)
        npt.assert_almost_equal(madeup.outputs.enp, 6.9, 1)
        npt.assert_almost_equal(madeup.outputs.residual_scale, 1.0868, 4)

    def test_2d_modnormalize(self):
        "2D - normalization modification"
        (x, y, results, _, _) = self.d
        madeup = loess(x, y)
        madeup.model.span = 0.8
        madeup.model.drop_square = [True, False]
        madeup.model.parametric = [True, False]
        madeup.model.family = "symmetric"
        madeup.model.normalize = False
        madeup.fit()
        npt.assert_almost_equal(madeup.outputs.fitted_values, results[3], 5)
        npt.assert_almost_equal(madeup.outputs.enp, 6.9, 1)
        npt.assert_almost_equal(madeup.outputs.residual_scale, 1.0868, 4)

    def test_2d_pred_nostderr(self):
        "2D prediction - no stderr"
        (x, y, results, newdata1, _) = self.d
        madeup = loess(x, y)
        madeup.model.span = 0.5
        madeup.model.normalize = True
        prediction = madeup.predict(newdata1, stderror=False)
        npt.assert_almost_equal(prediction.values, results[4], 5)
        #
        prediction = madeup.predict(newdata1, stderror=False)
        npt.assert_almost_equal(prediction.values, results[4], 5)

    def test_2d_pred_nodata(self):
        "2D prediction - nodata"
        (x, y, _, _, _) = self.d
        madeup = loess(x, y)
        try:
            madeup.predict(None)
        except ValueError:
            pass
        else:
            raise AssertionError("The test should have failed")

    def test_2d_pred_stderr(self):
        "2D prediction - w/ stderr"
        (x, y, results, _, newdata2) = self.d
        madeup = loess(x, y)
        madeup.model.span = 0.5
        madeup.model.normalize = True
        prediction = madeup.predict(newdata2, stderror=True)
        npt.assert_almost_equal(prediction.values, results[5], 5)
        npt.assert_almost_equal(prediction.stderr, [0.276746, 0.278009], 5)
        npt.assert_almost_equal(prediction.residual_scale, 0.969302, 6)
        npt.assert_almost_equal(prediction.df, 81.2319, 4)

        # Direct access
        prediction = madeup.predict(newdata2, stderror=True)
        npt.assert_almost_equal(prediction.values, results[5], 5)
        npt.assert_almost_equal(prediction.stderr, [0.276746, 0.278009], 5)
        npt.assert_almost_equal(prediction.residual_scale, 0.969302, 6)
        npt.assert_almost_equal(prediction.df, 81.2319, 4)

    def test_2d_pred_confinv(self):
        "2D prediction - confidence"
        (x, y, results, _, newdata2) = self.d
        madeup = loess(x, y)
        madeup.model.span = 0.5
        madeup.model.normalize = True
        prediction = madeup.predict(newdata2, stderror=True)
        ci = prediction.confidence(alpha=0.01)
        npt.assert_almost_equal(ci.lower, results[6][::3], 5)
        npt.assert_almost_equal(ci.fit, results[6][1::3], 5)
        npt.assert_almost_equal(ci.upper, results[6][2::3], 5)


class TestLoessGas(object):
    "Test class for lowess."

    d = gas_data()

    def test_1dbasic(self):
        "Basic test 1d"
        (E, NOx, _, _, _, results) = self.d
        gas = loess(E, NOx)
        gas.model.span = 2./3.
        gas.fit()
        npt.assert_almost_equal(gas.outputs.fitted_values, results[0], 6)
        npt.assert_almost_equal(gas.outputs.enp, 5.5, 1)
        npt.assert_almost_equal(gas.outputs.residual_scale, 0.3404, 4)

    def test_1dbasic_alt(self):
        "Basic test 1d - part #2"
        (E, NOx, _, _, _, results) = self.d
        gas_null = loess(E, NOx)
        gas_null.model.span = 1.0
        gas_null.fit()
        npt.assert_almost_equal(gas_null.outputs.fitted_values, results[1], 6)
        npt.assert_almost_equal(gas_null.outputs.enp, 3.5, 1)
        npt.assert_almost_equal(gas_null.outputs.residual_scale, 0.5197, 4)

    def test_1dpredict(self):
        "Basic test 1d - prediction"
        (E, NOx, gas_fit_E, _, _, results) = self.d
        gas = loess(E, NOx, span=2./3.)
        gas.fit()
        predicted = gas.predict(gas_fit_E, stderror=False).values
        npt.assert_almost_equal(predicted, results[2], 6)

    def test_1dpredict_2(self):
        "Basic test 1d - new predictions"
        (E, NOx, _, newdata, _, results) = self.d
        # gas = loess(E, NOx, span=2./3.)
        gas = loess(E, NOx)
        gas.model.span = 2./3.
        prediction = gas.predict(newdata, stderror=True)
        ci = prediction.confidence(alpha=0.01)
        npt.assert_almost_equal(ci.lower, results[3][0::3], 6)
        npt.assert_almost_equal(ci.fit, results[3][1::3], 6)
        npt.assert_almost_equal(ci.upper, results[3][2::3], 6)

    def test_anova(self):
        "Tests anova"
        (E, NOx, _, _, _, results) = self.d
        gas = loess(E, NOx, span=2./3.)
        gas.fit()
        gas_null = loess(E, NOx, span=1.0)
        gas_null.fit()
        gas_anova = loess_anova(gas, gas_null)
        gas_anova_theo = results[4]
        npt.assert_almost_equal(gas_anova.dfn, gas_anova_theo[0], 5)
        npt.assert_almost_equal(gas_anova.dfd, gas_anova_theo[1], 5)
        npt.assert_almost_equal(gas_anova.F_value, gas_anova_theo[2], 5)
        npt.assert_almost_equal(gas_anova.Pr_F, gas_anova_theo[3], 5)

    def test_failures(self):
        "Tests failures"
        (E, NOx, gas_fit_E, _, _, _) = self.d
        gas = loess(E, NOx, span=2./3.)
        # This one should fail (all parametric)
        gas.model.parametric = True
        with pytest.raises(ValueError):
            gas.fit()

        # This one also (all drop_square)
        gas.model.drop_square = True
        with pytest.raises(ValueError):
            gas.fit()

        gas.model.degree = 1
        with pytest.raises(ValueError):
            gas.fit()

        # This one should not (revert to std)
        gas.model.parametric = False
        gas.model.drop_square = False
        gas.model.degree = 2
        gas.fit()

        # Now, for predict .................
        prediction = gas.predict(gas_fit_E, stderror=False)
        # This one should fail (extrapolation & blending)
        with pytest.raises(ValueError):
            gas.predict(prediction.values, stderror=False)

        # But this one should not ..........
        gas.predict(gas_fit_E, stderror=False)
