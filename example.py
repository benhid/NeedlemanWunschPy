from NeedlemanWunschPy.algorithms import NeedlemanWunschLinear
from NeedlemanWunschPy.utils import read_fasta_as_a_list_of_pairs, totally_conserved_columns
from NeedlemanWunschPy.substitutionmatrix import Blosum50

# Initialization
seqA = read_fasta_as_a_list_of_pairs("data/test1.fasta")[0][1]
seqB = read_fasta_as_a_list_of_pairs("data/test2.fasta")[0][1]
gap_penalty = -2

# Set to False if we don't want
save_score_matrix_to_file = True

# Create the alignment
seqAaln, seqBaln = NeedlemanWunschLinear(seqA, seqB, gap_penalty, Blosum50()) \
    .get_alignment(save_score_matrix_to_file)

# Save results to file
with open('output/traceback.txt', 'w') as output:
    output.write('[SEQUENCE1] ' + seqAaln + '\n' +
                 '[CONSERVED] ' + totally_conserved_columns(seqAaln, seqBaln) + '\n' +
                 '[SEQUENCE2] ' + seqBaln)