from unittest import TestCase, mock

import numpy as np
from scipy import stats

from esdt import dvc


class TestLoss(TestCase):

    def test_correct_is_less_than_incorrect(self):
        dprimes = np.ones(2, 'd')
        criteria = np.array([0.25, 0.75])
        pA = stats.norm(0.5).cdf(0.25)
        pB = stats.norm(0.5).cdf(0.75)

        # We can do this, because we assume independence
        responses = dvc.ResponseCollection(pA*pA, pA*pB, pB*pA, pB*pB)

        args = (responses, dprimes, criteria)

        print(dvc.loss(0, *args))
        self.assertLess(dvc.loss(0, *args), dvc.loss(.2, *args))
