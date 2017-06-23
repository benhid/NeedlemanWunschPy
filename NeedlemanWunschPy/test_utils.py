import unittest

from NeedlemanWunschPy.utils import totally_conserved_columns, read_fasta_as_a_list_of_pairs


class TotallyConservedColumnsTestCases(unittest.TestCase):

    def setUp(self):
        pass

    def test_should_two_identical_sequences_return_100_percent_conserved_columns(self):
        result = totally_conserved_columns("AA","AA")
        expected = "||"
        self.assertEqual(result, expected)

    def test_should_two_sequences_of_different_lenght_raise_exception(self):
        with self.assertRaises(Exception):
            totally_conserved_columns("ACGT", "ACG")


class ReadFastaAsListOfPairsTestCases(unittest.TestCase):

    def setUp(self):
        print("setUp: TEST BEGIN")
        self.test_file = "test1.fasta"
        self.test2_file = "test2.fasta"

    def tearDown(self):
        print("tearDown: ENDED")

    def test_read_fasta_of_only_1_secuence_as_list_of_pairs(self):
        result = read_fasta_as_a_list_of_pairs(self.test2_file)
        expected = [['EXAMPLE_1_SEQUENCE', 'ACCGGT']]
        self.assertEqual(result, expected)

    def test_read_fasta_as_list_of_pairs(self):
        result = read_fasta_as_a_list_of_pairs(self.test_file)
        expected = [['EXAMPLE_SEQUENCE', 'ACGT'], ['OTHER', 'AACCGGTT']]
        self.assertEqual(result, expected)

    def test_file_not_found_list(self):
        with self.assertRaises(Exception):
            read_fasta_as_a_list_of_pairs(".txt")


if __name__ == '__main__':
    unittest.main()
