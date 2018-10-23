from unittest import TestCase
import doctest


class TestReadme(TestCase):

    def test_examples(self):
        doctest.testfile('../../../README.md', optionflags=doctest.ELLIPSIS)
