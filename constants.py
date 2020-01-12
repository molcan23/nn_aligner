import numpy as np

START = "start"
END = "end"
BASES = "bases"
BASE_KIND = "base_kind"
POSITION = "position"
LENGTH = 15
NN_VECTOR_LENGTH = 101
KMER_LENGTH = 6
VALUE = 'value'
STD = 'std'
SIGNAL_LENGTH = 30
DNA_BASES = ('A', 'C', 'T', 'G')
DNA = {
    'A': np.array([0, 0, 0, 1]),
    'C': np.array([0, 0, 1, 0]),
    'G': np.array([0, 1, 0, 0]),
    'T': np.array([1, 0, 0, 0])
}
NUMBER_OF_EXAMPLES = 10000
