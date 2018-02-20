from unittest import TestCase, mock

import numpy as np
from scipy import stats

from esdt import sdt


class TestBasicSDT(TestCase):

    def test_analytical(self):
        dprime = 1
        gamma = 0.2
        hit_rate = 1 - stats.norm(0.5*dprime).cdf(gamma)
        false_alarm_rate = 1 - stats.norm(-0.5*dprime).cdf(gamma)
        d, g = sdt.basic_sdt(hit_rate, false_alarm_rate)

        self.assertAlmostEqual(dprime, d)
        self.assertAlmostEqual(gamma, g)

    @mock.patch('esdt.sdt.basic_sdt')
    def test_hits_fa(self, mock_basic):
        targets = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        respons = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
        sdt.basic_sdt_analysis(targets, respons)
        mock_basic.assert_called_once_with(0.6, 0.2)

    def test_hits1(self):
        targets = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        respons = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        d, g = sdt.basic_sdt_analysis(targets, respons)
        self.assertLess(d, np.inf)
        self.assertLess(g, np.inf)

    def test_fa1(self):
        targets = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        respons = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
        d, g = sdt.basic_sdt_analysis(targets, respons)
        self.assertLess(d, np.inf)
        self.assertLess(g, np.inf)


class TestLoss(TestCase):

    def setUp(self):
        self.dprimes = np.ones(2, 'd')
        self.criteria = np.array([0.25, 0.75])

    def test_correct_is_less_than_incorrect(self):
        pA = stats.norm(0.5*self.dprimes[0]).cdf(self.criteria[0])
        pB = stats.norm(0.5*self.dprimes[1]).cdf(self.criteria[1])

        # We can do this, because we assume independence
        responses = sdt.ResponseCollection(pA*pA, pA*pB, pB*pA, pB*pB)

        args = (responses, self.dprimes, self.criteria)

        self.assertLess(sdt.loss(0, *args), sdt.loss(.2, *args))

    def test_correct_is_less_than_incorrect_other(self):
        pA = stats.norm(0.5*self.dprimes[0]).cdf(self.criteria[0])
        pB = stats.norm(0.5*self.dprimes[1]).cdf(self.criteria[1])

        # We can do this, because we assume independence
        responses = sdt.ResponseCollection(pA*pA, pA*pB, pB*pA, pB*pB)

        args = (responses, self.dprimes, self.criteria)

        self.assertLess(sdt.loss(0, *args), sdt.loss(-.2, *args))

    @mock.patch('esdt.sdt.dvc_from_probs')
    def test_dvc_for_category(self, mock_from_probs):
        responses_1 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        responses_2 = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

        sdt.dvc_for_category(
            responses_1, responses_2, self.dprimes, self.criteria)

        expected_responses = sdt.ResponseCollection(0.3, .2, .2, .3)

        mock_from_probs.assert_called_once_with(
            expected_responses, self.dprimes, self.criteria)

    @mock.patch('scipy.optimize.fmin')
    def test_dvc_from_probs(self, mock_optimize):
        mock_optimize.return_value = 'ANY_CORRELATION'
        r = sdt.dvc_from_probs('ANY_RESPONSES', 'ANY_DPRIMES', 'ANY_CRITERIA')
        mock_optimize.assert_called_once_with(
            sdt.loss,
            [0.],
            args=('ANY_RESPONSES',
                  np.asarray('ANY_DPRIMES'),
                  np.asarray('ANY_CRITERIA')),
            disp=0
        )
        self.assertEqual(r, 'ANY_CORRELATION')
