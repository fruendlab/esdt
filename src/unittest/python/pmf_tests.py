from unittest import TestCase, mock

import numpy as np
from scipy import stats

from esdt import pmf


class TestPsychometricFunction(TestCase):

    def setUp(self):
        self.data = np.array([[0.4, 0.2, 5],
                              [0.8, 0.6, 5],
                              [1.2, 1.0, 5],
                              [1.3, 0.8, 5]])

    def test_log_likelihood(self):
        patch_psi = mock.patch('esdt.pmf.PsychometricFunction.predict')
        with patch_psi as mock_psi:
            mock_psi.return_value = .5
            ps = pmf.PsychometricFunction(lambda x, t, w: x)
            ps.data = self.data
            ps.negloglikelihood('ANY_PARAM', 1, 2, 3)

            self.assertEqual(len(mock_psi.mock_calls), 1)
            self.assertTrue(
                np.all(1 == mock_psi.mock_calls[0][1][0]))
            self.assertEqual(mock_psi.mock_calls[0][1][1], 'ANY_PARAM')

    def test_posterior(self):
        ps = pmf.PsychometricFunction(lambda x, t, w: x, guess=0.5)
        ps.data = self.data
        p = ps.posterior(np.array([[.1, .1, .1]]).T)

        self.assertAlmostEqual(1., p.max())

    @mock.patch('esdt.pmf.stats.linregress')
    def test_get_start_uses_linear_regression(self, mock_linregress):
        mock_linregress.return_value = (1, 2, 'other')
        ps = pmf.PsychometricFunction(pmf.logistic, 0.5)
        start = ps.get_start(self.data[:, 0], self.data[:, 1])
        self.assertEqual(len(mock_linregress.mock_calls), 1)
        self.assertAlmostEqual(start[0], -2)
        self.assertAlmostEqual(start[1], 2)

    def test_fit_assigns_by_default(self):
        ps = pmf.PsychometricFunction(pmf.logistic, 0.5)
        self.assertIsNone(ps.params)
        params, ll = ps.fit(self.data)
        self.assertIsNotNone(ps.params)
        self.assertIsNotNone(ps.data)
        self.assertEqual(len(params), 3)
        self.assertGreater(ll, 0)

    def test_fit_does_not_assign_if_requested(self):
        ps = pmf.PsychometricFunction(pmf.logistic, 0.5)
        self.assertIsNone(ps.params)
        ps.fit(self.data, assign=False)
        self.assertIsNone(ps.params)
        self.assertIsNone(ps.data)

    @mock.patch('esdt.pmf.optimize.fmin')
    def test_fit_uses_starting_values_if_specified(self, mock_fmin):
        mock_fmin.return_value = np.ones(3, 'd')
        ps = pmf.PsychometricFunction(pmf.logistic, 0.5)
        ps.fit(self.data, start='ANY_STARTING_VALUE')
        self.assertEqual(mock_fmin.mock_calls[0][1][1], 'ANY_STARTING_VALUE')

    def test_jackknife_sem_drops_every_block(self):
        ps = pmf.PsychometricFunction(pmf.logistic, 0.5)
        ps.fit = mock.Mock()
        ps.fit.side_effect = [(k + np.zeros(3, 'd'), 1) for k in range(5)]
        ps.params = np.zeros(3, 'd')  # Needed for influence
        ps.jackknife_sem(self.data)
        self.assertEqual(len(ps.fit.mock_calls), self.data.shape[0])

    def test_deviance_test(self):
        ps = pmf.PsychometricFunction(pmf.logistic, 0.5)
        ps.params = np.array([1., 1., .03])
        p, D = ps.deviance_test(self.data, nsamples=20)
        self.assertSequenceEqual(D.shape, (20,))
        self.assertIsInstance(p, float)
        self.assertGreaterEqual(p, 0)
        self.assertLessEqual(p, 1)

    def test_transform_expresses_threshold_transformed(self):
        ps = pmf.PsychometricFunction(pmf.gumbel, 0.5, np.log10)
        self.data[:, 0] = [.01, .05, .1, .5]
        params, ll = ps.fit(self.data)
        self.assertGreater(params[0], -100)
        self.assertLess(params[0], np.log10(0.5))

    def test_transform_results_in_predictions_on_transformed_values(self):
        mock_F = mock.Mock(return_value=0.5)
        mock_log = mock.Mock()
        ps = pmf.PsychometricFunction(mock_F, 0.5, mock_log)
        ps.predict('ANY_X_VALUE', ['ANY_THRES', 'ANY_WIDTH', .05])
        mock_F.assert_called_once_with(mock_log.return_value,
                                       'ANY_THRES',
                                       'ANY_WIDTH')
        mock_log.assert_called_once_with('ANY_X_VALUE')


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


class TestSigmoids(TestCase):

    sigmoids = [pmf.gumbel, pmf.logistic]

    def test_at_neg_infty(self):
        for sigmoid in self.sigmoids:
            with self.subTest(sigmoid=sigmoid):
                self.assertAlmostEqual(sigmoid(-100, 0, 1), 0)

    def test_at_plus_infty(self):
        for sigmoid in self.sigmoids:
            with self.subTest(sigmoid=sigmoid):
                self.assertAlmostEqual(sigmoid(100, 0, 1), 1)

    def test_at_zero(self):
        for sigmoid in self.sigmoids:
            with self.subTest(sigmoid=sigmoid):
                self.assertAlmostEqual(sigmoid(0, 0, 1), 0.5)


class TestDeviance(TestCase):

    def test_should_be_zero_for_perfect_fit(self):
        k = np.ones(4, 'd')
        n = np.ones(4, 'd') * 2
        p = np.array([0.5]*4)
        self.assertAlmostEqual(pmf.deviance(k, n, p), 0)

    def test_should_work_with_integers(self):
        k = 1
        n = 2
        p = 0.5
        self.assertAlmostEqual(pmf.deviance(k, n, p), 0)

    def test_should_be_gt_0_for_imperfect_fit(self):
        k = np.array([1, 2, 3, 4], 'd')
        n = np.array([5, 5, 5, 5], 'd')
        p = np.array([.5]*4)
        self.assertGreater(pmf.deviance(k, n, p), 0)

    def test_should_broadcast_with_arrays(self):
        k = np.ones((20, 4), 'd')
        n = np.ones(4, 'd') * 2
        p = np.array([.5]*4)
        D = pmf.deviance(k, n, p, axis=1)
        self.assertAlmostEqual(D.max(), 0)
        self.assertSequenceEqual(D.shape, (20,))

    def test_should_work_with_integer_arrays(self):
        k = np.ones(4, 'i')
        n = np.ones(4, 'i') * 2
        p = np.array([0.5]*4)
        self.assertAlmostEqual(pmf.deviance(k, n, p), 0)
