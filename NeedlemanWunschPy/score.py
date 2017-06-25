class SumOfPairs:
    """ Class for returning the alignment score of 2 sequences given the substitution matrix. """

    def __init__(self, substitution_matrix):
        self.substitution_matrix = substitution_matrix

    def compute(self, seqA, seqB):
        length_sequence = len(seqA)

        final_score = 0

        for k in range(length_sequence):
            final_score += self.get_score_of_two_chars(seqA[k], seqB[k])

        return final_score

    def get_score_of_two_chars(self, charA, charB):
        return self.substitution_matrix.get_score(charA, charB)