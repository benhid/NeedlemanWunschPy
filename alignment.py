#!/usr/bin/env python

import logging.config
import time
import os

import numpy as np
import pandas as pd
from Bio.SubsMat import MatrixInfo

# About
__author__ = "Antonio Benitez Hidalgo"
__email__ = "antonio.b@uma.es"
__version__ = "1.3-SNAPSHOT"

# Load logger config from file
logging.config.fileConfig("logconfig.ini")
logger = logging.getLogger(__name__)


"""
 Global alignment with simple gap costs using the Needleman-Wunsch algorithm.

 Note: The pseudo-code implemented in this example can be found here:
            http://www.inf.fu-berlin.de/lehre/WS05/aldabi/downloads/pairAlign_part1.pdf
 More info: https://ab.inf.uni-tuebingen.de/teaching/ws06/albi1/script/pairalign_script.pdf,
            http://www.itu.dk/people/sestoft/bsa/graphalign.html
"""


def get_time_of_execution(f):
    """ Decorator to get time of execution """

    def wrapped(*args, **kwargs):
        start_time = time.time()
        res = f(*args, **kwargs)
        logger.info("Time elapsed to " + f.__name__ + " (s): " + str(time.time() - start_time))
        return res

    return wrapped


class NeedlemanWunschLinear():
    def __init__(self, seqA, seqB, gap_penalty = 8, substitution_matrix = MatrixInfo.blosum50):
        self.seq_h = seqA  # seq A (horizontal, rows)
        self.seq_v = seqB  # seq B (vertical, columns)
        self.gap_penalty = gap_penalty  # gap penalty both for opening and extension (linear gap penalty)
        self.substitution_matrix = substitution_matrix

        self.num_rows = len(seqA) + 1  # number of rows plus one (the first one)
        self.num_cols = len(seqB) + 1  # number of columns plus one (the first one)

        # Initialization of scoring matrix
        self.M = np.zeros(shape=(self.num_cols, self.num_rows),
                          dtype=np.int)  # matrix of zeros of integers (score matrix)

    def _get_score(self, i, j):
        """ Get score from two chars.

        :return: Score of the pair of chars.
        """

        char1, char2 = self.seq_h[j-1], self.seq_v[i-1]

        if char1 is '-' and char2 is '-':
            score = 1
        elif char1 is '-' or char2 is '-':
            score = self.gap_penalty
        else:
            # Note that the substitution matrix is triangular
            if (char1, char2) in self.substitution_matrix:
                score = self.substitution_matrix[char1, char2]
            else:
                score = self.substitution_matrix[char2, char1]

        return score

    @get_time_of_execution
    def _compute_score_matrix(self):
        """ Needleman Wunsch algorithm. Formula:
                M(0,0) = 0
                M(i,0) = M(i-1,0) - gap_penalty
                M(0,j) = M(0,j-1) - gap_penalty
                M(i,j) = max{ M(i-1,j-1) + score , M(i-1,j) - gap_penalty, M(i,j-1) - gap_penalty }

            M will be our scoring matrix.
        """

        # First column and row
        self.M[0, 0] = 0
        for j in range(self.num_cols):  # First row
            self.M[j, 0] = self.M[j-1, 0] - self.gap_penalty
        for i in range(self.num_rows):  # First column
            self.M[0, i] = self.M[0, i-1] - self.gap_penalty

        # Rest of the matrix (recursive)
        for i in range(1, self.num_cols):
            for j in range(1, self.num_rows):
                # current_score = self.get_score(i, j)
                # score_diagonal = self.M[i - 1, j - 1] + current_score
                # score_up = self.M[i - 1, j] - self.gap_penalty
                # score_left = self.M[i, j - 1] - self.gap_penalty

                self.M[i, j] = max(self.M[i-1, j-1] + self._get_score(i, j),
                                   self.M[i-1, j] - self.gap_penalty,
                                   self.M[i, j-1] - self.gap_penalty)

    @get_time_of_execution
    def _save_matrix_to_file(self, matrix, filename):
        """ Save matrix dataframe into a .csv file separated by commas.

        :param matrix: Numpy matrix.
        :param filename: Name of the file.
        """

        data = pd.DataFrame(matrix, index=list(' ' + self.seq_v), columns=list(' ' + self.seq_h))

        # Save to output directory
        os.makedirs(os.path.dirname("output/"), exist_ok=True)
        data.to_csv("output/"+filename+'.csv', sep=',', encoding='utf8')

    @get_time_of_execution
    def _traceback(self):
        """ Traceback algorithm. We can make the traceback looking to the traceback matrix (T):
            ...up arrow: we consume a character from the vertical sequence and add a gap to the horizontal one
            ...left arrow: we consume a character from the horizontal sequence and add a gap to the vertical one
            ...diagonal arrow: we consume a character from both sequences.
        """

        alnseqA, alnseqB, path = [], [], []

        i, j = self.num_cols - 1, self.num_rows - 1

        while i > 0 and j > 0:
            # score_diagonal = self.M[i - 1, j - 1] + self.get_score(i, j)
            # score_up = self.M[i - 1, j] - self.gap_penalty
            # score_left = self.M[i, j - 1] - self.gap_penalty
            # max_score = max(score_diagonal, score_up, score_left)

            if self.M[i,j] == self.M[i-1, j-1] + self._get_score(i, j):
                logger.debug(" > going diagonal, i={0}, j={1}".format(i, j))
                alnseqA.append(self.seq_v[i-1])
                alnseqB.append(self.seq_h[j-1])
                path.append(self.M[i, j])
                i, j = i-1, j-1
            elif self.M[i,j] == self.M[i-1, j] - self.gap_penalty:
                logger.debug(" > going up, i={0}, j={1}".format(i, j))
                alnseqA.append(self.seq_v[i-1])
                alnseqB.append('-')
                path.append(self.M[i, j])
                i -= 1
            else:
                logger.debug(" > going left, i={0}, j={1}".format(i, j))
                alnseqA.append('-')
                alnseqB.append(self.seq_h[j-1])
                path.append(self.M[i, j])
                j -= 1

        # Finish tracing up to the top left cell
        while i > 0:
            logger.debug(" > going left, i={0}, j={1}".format(i, j))
            alnseqA.append(self.seq_v[i-1])
            alnseqB.append('-')
            path.append(self.M[i, j])
            i -= 1
        while j > 0:
            logger.debug(" > going up, i={0}, j={1}".format(i, j))
            alnseqA.append('-')
            alnseqB.append(self.seq_h[j-1])
            path.append(self.M[i, j])
            j -= 1

        # Return both reversed alignment sequences
        seqaln_h, seqaln_v = "".join(alnseqA)[::-1], "".join(alnseqB)[::-1]
        logger.debug('Pathway chosen in the traceback matrix: {0}'.format(path))

        return seqaln_h, seqaln_v

    @get_time_of_execution
    def get_alignment(self, save_score_matrix_to_file = False):
        self._compute_score_matrix()

        if save_score_matrix_to_file:
            self._save_matrix_to_file(self.M, 'scoring_matrix')

        return self._traceback()

@get_time_of_execution
def read_fasta_as_a_list_of_pairs(filename):
    try:
        f = open(filename,'r', encoding="utf8")
    except:
        logger.error('File not found', exc_info=True)
        raise FileNotFoundError

    seq = None
    id = None
    list = []

    for line in f:
        if line.strip():
            if line.startswith('>'):
                if id is not None:
                    list.append([id, seq])

                id = line[1:].strip()
                seq = ''
            else:
                seq += line.strip()

    list.append([id, seq])

    f.close()
    return list

@get_time_of_execution
def totally_conserved_columns(seqA, seqB):
    """ Method for getting the position of the totally conserved columns.

    :return: String with vertical bars in those places were the columns are totally conserved.
    """

    if len(seqA) != len(seqB):
        logger.error('Different lengths!', exc_info=True)
        raise Exception

    totally_conserved_columns = []

    for k in range(len(seqA)):
        if seqA[k] == seqB[k]:
            totally_conserved_columns.append('|')
        else:
            totally_conserved_columns.append(' ')

    return "".join(totally_conserved_columns)


if __name__ == '__main__':
    # Initialization
    seqA = read_fasta_as_a_list_of_pairs("data/test1.fasta")
    seqB = read_fasta_as_a_list_of_pairs("data/test2.fasta")

    gap_penalty = 8  # in this case, gap penalty must be a positive int
    save_score_matrix_to_file = True

    # Create the alignment
    seqAaln, seqBaln = NeedlemanWunschLinear(seqA[0][1], seqB[0][1], gap_penalty) \
        .get_alignment(save_score_matrix_to_file)

    # Save to file
    with open('output/traceback.txt', 'w') as output:
        output.write('[SEQUENCE1] ' + seqAaln + '\n' +
                     '[CONSERVED] ' + totally_conserved_columns(seqAaln, seqBaln) + '\n' +
                     '[SEQUENCE2] ' + seqBaln)
