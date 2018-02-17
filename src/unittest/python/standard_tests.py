from unittest import TestCase, mock

from scipy import stats

from esdt import standard


class TestBasicSDT(TestCase):

    def test_analytical(self):
        dprime = 1
        gamma = 0.2
        hit_rate = 1 - stats.norm(0.5*dprime).cdf(gamma)
        false_alarm_rate = 1 - stats.norm(-0.5*dprime).cdf(gamma)
        d, g = standard.basic_sdt(hit_rate, false_alarm_rate)

        self.assertAlmostEqual(dprime, d)
        self.assertAlmostEqual(gamma, g)

    @mock.patch('esdt.standard.basic_sdt')
    def test_hits_fa(self, mock_basic):
        targets = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        respons = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
        standard.basic_sdt_analysis(targets, respons)
        mock_basic.assert_called_once_with(0.6, 0.2)
