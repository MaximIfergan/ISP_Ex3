import unittest
import numpy as np
from ctc import ctc_loss

class TestCTCLoss(unittest.TestCase):

    def test_simple_case(self):
        y = np.array([
            [0.6, 0.2, 0.2],
            [0.3, 0.4, 0.3],
            [0.1, 0.1, 0.8]
        ])
        p = "ab"
        alphabet = "ab"
        expected = 0.0608
        result = ctc_loss(y, p, alphabet)
        self.assertAlmostEqual(result, expected, places=4)

    def test_blank_only(self):
        y = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ])
        p = ""
        alphabet = "ab"
        expected = 1.0
        result = ctc_loss(y, p, alphabet)
        self.assertEqual(result, expected)

    def test_no_match(self):
        y = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        p = "b"
        alphabet = "ab"
        expected = 0.0
        result = ctc_loss(y, p, alphabet)
        self.assertEqual(result, expected)

    def test_longer_sequence(self):
        y = np.array([
            [0.3, 0.3, 0.4],
            [0.2, 0.5, 0.3],
            [0.1, 0.7, 0.2],
            [0.6, 0.2, 0.2],
            [0.3, 0.4, 0.3]
        ])
        p = "aba"
        alphabet = "ab"
        expected = 0.02214
        result = ctc_loss(y, p, alphabet)
        self.assertAlmostEqual(result, expected, places=5)

    def test_repeated_characters(self):
        y = np.array([
            [0.4, 0.3, 0.3],
            [0.2, 0.6, 0.2],
            [0.1, 0.8, 0.1],
            [0.3, 0.4, 0.3]
        ])
        p = "aab"
        alphabet = "ab"
        expected = 0.02592
        result = ctc_loss(y, p, alphabet)
        self.assertAlmostEqual(result, expected, places=5)

if __name__ == '__main__':
    unittest.main()