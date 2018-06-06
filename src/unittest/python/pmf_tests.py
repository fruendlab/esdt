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
        p = ps.posterior(np.array([[.1, .1, .1]]))

        self.assertAlmostEqual(1., p.max())
        self.assertAlmostEqual(1., p.mean())


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
