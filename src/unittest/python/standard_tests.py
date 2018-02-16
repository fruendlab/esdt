from unittest import TestCase

from scipy import stats

from esdt.standard import basic_sdt


class TestBasicSDT(TestCase):

    def test_analytical(self):
        dprime = 1
        gamma = 0.2
        hit_rate = 1 - stats.norm(0.5*dprime).cdf(gamma)
        false_alarm_rate = 1 - stats.norm(-0.5*dprime).cdf(gamma)
        d, g = basic_sdt(hit_rate, false_alarm_rate)

        self.assertAlmostEqual(dprime, d)
        self.assertAlmostEqual(gamma, g)
