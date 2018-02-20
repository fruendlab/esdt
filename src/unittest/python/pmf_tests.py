from unittest import TestCase, mock

import numpy as np
from scipy import stats

from esdt import pmf


class TestPsychometricFunction(TestCase):

    def setUp(self):
        self.data = np.array([[0.4, 1, 5],
                              [0.8, 3, 5],
                              [1.2, 5, 5],
                              [1.3, 4, 5]])

    def test_log_likelihood(self):
        patch_psi = mock.patch('esdt.pmf.PsychometricFunction.psi')
        patch_priors = mock.patch('esdt.pmf.PsychometricFunction.priors',
                                  new_callable=mock.PropertyMock)
        with patch_psi as mock_psi, patch_priors as mock_priors:
            mock_psi.return_value = .5
            ps = pmf.PsychometricFunction(self.data)
            ll = ps.loglikelihood('ANY_PARAM')

            self.assertEqual(len(mock_psi.mock_calls), 1)
            self.assertTrue(
                np.all(self.data[:, 0] == mock_psi.mock_calls[0][1][0]))
            self.assertEqual(mock_psi.mock_calls[0][1][1], 'ANY_PARAM')

            mock_priors.assert_not_called()
            B = stats.binom(self.data[:, 2], 0.5)
            self.assertAlmostEqual(
                float(ll),
                float(B.logpmf(self.data[:, 1]).sum()))

    def test_posterior(self):
        patch_ll = mock.patch('esdt.pmf.PsychometricFunction.loglikelihood')
        patch_priors = mock.patch('esdt.pmf.PsychometricFunction.priors',
                                  new_callable=mock.PropertyMock)
        with patch_ll as mock_ll, patch_priors as mock_priors:
            mock_priors.return_value = [stats.norm(0, 1)]
            mock_ll.return_value = np.ones(5, 'd')
            ps = pmf.PsychometricFunction(self.data)
            p = ps.posterior([1.])

            mock_ll.assert_called_once_with([1.])

            self.assertAlmostEqual(1., p.max())
            self.assertAlmostEqual(1., p.mean())

    def test_abstract_psi(self):
        ps = pmf.PsychometricFunction('ANY_DATA')
        with self.assertRaises(NotImplementedError):
            ps.psi('ANY_STIMULUS',
                   'ANY_PARAMETERS')

    def test_abstract_get_thres(self):
        ps = pmf.PsychometricFunction('ANY_DATA')
        with self.assertRaises(NotImplementedError):
            ps.get_thres('ANY_STIMULUS',
                         'ANY_PARAMETERS')

    def test_abstract_get_slope(self):
        ps = pmf.PsychometricFunction('ANY_DATA')
        with self.assertRaises(NotImplementedError):
            ps.get_slope('ANY_STIMULUS',
                         'ANY_PARAMETERS')

    def test_abstract_priors(self):
        ps = pmf.PsychometricFunction('ANY_DATA')
        with self.assertRaises(NotImplementedError):
            ps.priors


class TestGrid(TestCase):

    def test_dims(self):
        grd = pmf.mkgrid((-10, 10, 5))
        self.assertSequenceEqual(grd.shape, (1, 5))

    def test_more_dims(self):
        grd = pmf.mkgrid((-10, 10, 5), (-10, 10, 5), (1e-5, .2, 5))
        self.assertSequenceEqual(grd.shape, (3, 5, 5, 5))


class TestIntegration(TestCase):

    def test_integration(self):
        mock_pmf = mock.Mock()
        mock_pmf.posterior = mock.Mock(return_value=np.ones(10, 'd'))
        grid = np.ones((3, 5, 5, 5), 'd')
        g, p = pmf.integrate_posterior(mock_pmf, grid)
        self.assertSequenceEqual(g.shape, (3, 5*5*5))
        self.assertSequenceEqual(p.tolist(), [.1]*10)
        posterior_calls = mock_pmf.posterior.mock_calls
        self.assertEqual(len(posterior_calls), 1)
        np.testing.assert_array_almost_equal(posterior_calls[0][1][0],
                                             grid.reshape((3, -1)))

    def test_get_stats(self):
        x = np.mgrid[-10:10:20j]
        posterior = stats.norm(0, 1).pdf(x)
        posterior /= posterior.sum()
        m, sg = pmf.get_stats(posterior, x)
        self.assertAlmostEqual(m, 0, 6)
        self.assertAlmostEqual(sg, 1, 5)


class TestLogisticAfc(TestCase):

    def setUp(self):
        self.param = np.array([.5, .2, .03]).T

    def test_lower_bound(self):
        ps = pmf.LogisticAfc('ANY_DATA')
        self.assertAlmostEqual(float(ps.psi(-40, self.param)), 0.5, 3)

    def test_upper_bound(self):
        ps = pmf.LogisticAfc('ANY_DATA')
        self.assertAlmostEqual(float(ps.psi(40, self.param)), 0.97, 3)

    def test_get_thres(self):
        ps = pmf.LogisticAfc('ANY_DATA')
        th = ps.get_thres(self.param, 0.65)
        self.assertAlmostEqual(ps.psi(th, self.param), 0.65)

    def test_get_slope(self):
        ps = pmf.LogisticAfc('ANY_DATA')
        th = ps.get_thres(self.param, 0.65)
        sl = ps.get_slope(self.param, 0.65)
        eps = 1e-7
        sl_ = 0.5*(ps.psi(th + eps, self.param)
                   - ps.psi(th - eps, self.param))/eps
        self.assertAlmostEqual(float(sl), float(sl_))

    def test_priors(self):
        ps = pmf.LogisticAfc('ANY_DATA')
        priors = ps.priors
        self.assertEqual(len(priors), 3)
