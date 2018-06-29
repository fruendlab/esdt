from unittest import TestCase

from esdt import utils


class TestSafeLog(TestCase):

    def test_for_1_is_0(self):
        self.assertAlmostEqual(utils.safe_log(1), 0)

    def test_for_0_is_negative(self):
        self.assertLess(utils.safe_log(0), -5)

    def test_for_negative_is_like_0(self):
        self.assertAlmostEqual(utils.safe_log(-1), utils.safe_log(0))
