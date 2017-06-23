def read_fasta_as_a_list_of_pairs(filename):
    try:
        f = open(filename,'r', encoding="utf8")
    except:
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

def totally_conserved_columns(seqAaln, seqBaln):
    """ Method for getting the position of the totally conserved columns.

    :return: String with vertical bars in those places were the columns are totally conserved.
    """

    if len(seqAaln) != len(seqBaln):
        raise Exception

    totally_conserved_columns = []

    for k in range(len(seqAaln)):
        if seqAaln[k] == seqBaln[k]:
            totally_conserved_columns.append('|')
        else:
            totally_conserved_columns.append(' ')

    return "".join(totally_conserved_columns)
