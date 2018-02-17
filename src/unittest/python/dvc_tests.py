from unittest import TestCase, mock

import numpy as np
from scipy import stats

from esdt import dvc


class TestLoss(TestCase):

    def setUp(self):
        self.dprimes = np.ones(2, 'd')
        self.criteria = np.array([0.25, 0.75])

    def test_correct_is_less_than_incorrect(self):
        pA = stats.norm(0.5*self.dprimes[0]).cdf(self.criteria[0])
        pB = stats.norm(0.5*self.dprimes[1]).cdf(self.criteria[1])

        # We can do this, because we assume independence
        responses = dvc.ResponseCollection(pA*pA, pA*pB, pB*pA, pB*pB)

        args = (responses, self.dprimes, self.criteria)

        self.assertLess(dvc.loss(0, *args), dvc.loss(.2, *args))

    def test_correct_is_less_than_incorrect_other(self):
        pA = stats.norm(0.5*self.dprimes[0]).cdf(self.criteria[0])
        pB = stats.norm(0.5*self.dprimes[1]).cdf(self.criteria[1])

        # We can do this, because we assume independence
        responses = dvc.ResponseCollection(pA*pA, pA*pB, pB*pA, pB*pB)

        args = (responses, self.dprimes, self.criteria)

        self.assertLess(dvc.loss(0, *args), dvc.loss(-.2, *args))

    @mock.patch('esdt.dvc.dvc_from_probs')
    def test_dvc_for_category(self, mock_from_probs):
        responses_1 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        responses_2 = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

        dvc.dvc_for_category(
            responses_1, responses_2, self.dprimes, self.criteria)

        expected_responses = dvc.ResponseCollection(0.3, .2, .2, .3)

        mock_from_probs.assert_called_once_with(
            expected_responses, self.dprimes, self.criteria)

    @mock.patch('scipy.optimize.fmin')
    def test_dvc_from_probs(self, mock_optimize):
        mock_optimize.return_value = 'ANY_CORRELATION'
        r = dvc.dvc_from_probs('ANY_RESPONSES', 'ANY_DPRIMES', 'ANY_CRITERIA')
        mock_optimize.assert_called_once_with(
            dvc.loss,
            [0.],
            args=('ANY_RESPONSES', 'ANY_DPRIMES', 'ANY_CRITERIA'))
        self.assertEqual(r, 'ANY_CORRELATION')
