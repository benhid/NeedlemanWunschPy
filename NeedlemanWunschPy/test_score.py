import unittest

from NeedlemanWunschPy.substitutionmatrix import PAM250, Blosum62
from NeedlemanWunschPy.score import SumOfPairs


class SumOfPairsTestCases(unittest.TestCase):
    def setUp(self):
        print("setUp: RUNNING SubstutionMatrixTestCases")
        self.matrix_PAM250 = PAM250()
        self.matrix_Blosum62 = Blosum62()

    def tearDown(self):
        print("tearDown: TEST ENDED")

    def test_should_score_be_14(self):
        score = SumOfPairs(self.matrix_PAM250)
        seqA, seqB = 'ACGT', 'ACCT'

        result = score.compute(seqA, seqB)
        expected = 14
        self.assertEqual(result, expected)

    def test_should_score_of_two_identical_sequences_be_6(self):
        score = SumOfPairs(self.matrix_PAM250)
        seqA, seqB = 'AAA', 'AAA'

        result = score.compute(seqA, seqB)
        expected = 6
        self.assertEqual(result, expected)

    def test_should_score_of_two_identical_chars_be_2(self):
        score = SumOfPairs(self.matrix_PAM250)
        charA, charB = 'A', 'A'

        result = score.get_score_of_two_chars(charA, charB)
        expected = 2
        self.assertEqual(result, expected)

    def test_should_get_score_throw_an_exception_if_a_char_is_invalid(self):
        score = SumOfPairs(self.matrix_PAM250)

        with self.assertRaises(Exception):
            score.get_score_of_two_chars('J', 'A')


if __name__ == '__main__':
    unittest.main()
