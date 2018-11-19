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
        params, ll = ps.ml_fit(self.data)
        self.assertIsNotNone(ps.params)
        self.assertIsNotNone(ps.data)
        self.assertEqual(len(params), 3)
        self.assertGreater(ll, 0)

    def test_fit_does_not_assign_if_requested(self):
        ps = pmf.PsychometricFunction(pmf.logistic, 0.5)
        self.assertIsNone(ps.params)
        ps.ml_fit(self.data, assign=False)
        self.assertIsNone(ps.params)
        self.assertIsNone(ps.data)

    @mock.patch('esdt.pmf.optimize.fmin')
    def test_fit_uses_starting_values_if_specified(self, mock_fmin):
        mock_fmin.return_value = np.ones(3, 'd')
        ps = pmf.PsychometricFunction(pmf.logistic, 0.5)
        ps.ml_fit(self.data, start='ANY_STARTING_VALUE')
        self.assertEqual(mock_fmin.mock_calls[0][1][1], 'ANY_STARTING_VALUE')

    def test_fit_is_deprecated(self):
        ps = pmf.PsychometricFunction(pmf.logistic, 0.5)
        ps.ml_fit = mock.Mock()
        with self.assertWarns(DeprecationWarning):
            ps.fit('ANY_DATA', start='ANY_START')
        ps.ml_fit.assert_called_once_with('ANY_DATA', start='ANY_START')

    def test_jackknife_sem_drops_every_block(self):
        ps = pmf.PsychometricFunction(pmf.logistic, 0.5)
        ps.ml_fit = mock.Mock()
        ps.ml_fit.side_effect = [(k + np.zeros(3, 'd'), 1) for k in range(5)]
        ps.params = np.zeros(3, 'd')  # Needed for influence
        ps.jackknife_sem(self.data)
        self.assertEqual(len(ps.ml_fit.mock_calls), self.data.shape[0])


class TestGrid(TestCase):

    def test_dims(self):
        grd = pmf.mkgrid((-10, 10, 5))
        self.assertSequenceEqual(grd.shape, (1, 5))

    def test_more_dims(self):
        grd = pmf.mkgrid((-10, 10, 5), (-10, 10, 5), (1e-5, .2, 5))
        self.assertSequenceEqual(grd.shape, (3, 5, 5, 5))


class TestSampleGridpoints(TestCase):

    @mock.patch('esdt.pmf.np.random.multinomial')
    def test_shapes(self, mock_multinomial):
        mock_multinomial.return_value = np.eye(4)
        posterior = [.1, .1, .1, .7]
        idx = pmf.sample_gridpoints(4, posterior)
        self.assertSequenceEqual(idx.shape, (4,))
        mock_multinomial.assert_called_once_with(1, posterior, size=4)
        self.assertSequenceEqual(idx.tolist(), [0, 1, 2, 3])


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
