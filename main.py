import h5py
import numpy as np
import glob

import random
import constants as cs

ideal_signal_values = {}
standard_deviation = 0


def base_to_signal_mapping(grp):
    """
    original_tandem.py
    Map bases to corresponding part of signal.
    """

    position_in_signal = [0 for _ in range(5)]
    for i in range(1, len(grp)):
        position_in_signal += [i for _ in range(grp[i][5])]
        # position_in_signal += [grp[i][0] for _ in range(grp[i][5])]

    # print(position_in_signal)
    return position_in_signal


def normalized_signal(grp):
    """
    Normilze grp.
    """

    position_in_signal = [grp[0][0] for _ in range(5)]

    for i in range(1, len(grp)):
        position_in_signal += [grp[i][0] for _ in range(grp[i][5])]
    position_in_signal -= np.mean(position_in_signal)
    position_in_signal /= np.std(position_in_signal)

    return position_in_signal


def basecall_to_reference_mapping(pair_str, shift, shift_basecalled):
    """
    Referencia namapovana na basecall, vrati nam mapping a list 4ic kde mame zaciatok a koniec v ref a basecalle
    """

    mapping = []
    index = -1
    i = 0
    global_shift = 0
    reference_long_match_position = []

    while i < len(pair_str) - 1:
        if pair_str[i] == ':':
            j = i + 1
            num_str = ''
            while pair_str[j] in '0123456789' and j < len(pair_str):
                num_str += pair_str[j]
                j += 1
                if j >= len(pair_str):
                    break
            i = j

            if int(num_str) > 19:
                reference_long_match_position.append([shift + index + 1,
                                                      shift + index + int(num_str) + 1,
                                                      shift_basecalled + global_shift + index + 1,
                                                      shift_basecalled + global_shift + index + int(num_str)])

            for _ in range(int(num_str)):
                index += 1
                mapping.append(index)
            continue

        elif pair_str[i] == '*':
            index += 1
            mapping.append(index)
            i += 3
            continue

        elif pair_str[i] == '-':
            i += 1

            while pair_str[i] in 'acgt' and i < len(pair_str):
                global_shift -= 1
                i += 1
                index += 1
            continue

        elif pair_str[i] == '+':
            i += 1

            while pair_str[i] in 'acgt' and i < len(pair_str):
                global_shift += 1
                mapping.append(index)
                i += 1
            continue

    return mapping, reference_long_match_position


def refrence_sequence_from_interval(ref, contig_name, i, j):
    """
    Z referencie 'ref', contigu 'contig_name' vrati cast od pozicie i po j.
    """

    sequence = []
    file_sequence = open(ref, 'r')
    read_now = False

    while True:
        base = file_sequence.readline()
        if '>' in base:
            read_now = False
        if read_now is True:
            sequence.append(base.rstrip('\n'))
        if contig_name in base:
            read_now = True
        if base == '':
            break

    return list(''.join(sequence)[i:j-1])


def basecalled_tandem_from_interval(filename, x, y):
    """
    Vypise bazy z basecallu na poziciach [x:y]
    """
    f = h5py.File(filename, 'r')
    grp = str(np.array(f.get('/Analyses/Basecall_1D_000/BaseCalled_template/Fastq')))
    i = 1
    while not (grp[i] == 'n' and grp[i - 1] == '\\'):
        i += 1
    i += 1
    j = i + 2
    while not (grp[j] == 'n' and grp[j - 1] == '\\'):
        j += 1
    # print(i, j)
    # print(grp[130:160])
    # print(grp[144:147])
    grp = grp[i:j-1]
    # print('grp', grp)
    return grp[x:y]


def reference_to_signal_partial_mapping(rb_map_string, reference_location, read_location, contig_name,
                                        ref_start, bas_start):
    """
    Namapuje referenciu na signal pre konkretny riadok z out.txt
    """

    a, b = basecall_to_reference_mapping(rb_map_string, ref_start, bas_start)
    f = h5py.File(read_location, 'r')
    grp = np.array(f.get('/Analyses/Basecall_1D_000/BaseCalled_template/Events'))
    bts = base_to_signal_mapping(grp)
    norm_sig = normalized_signal(grp)
    vectors_for_nn = np.array([], dtype=np.int64).reshape(0, cs.NN_VECTOR_LENGTH)

    for i in b:
        rs = i[0]
        re = i[1]
        bs = i[2]
        # R=B cast sekvencie
        ref = refrence_sequence_from_interval(reference_location, contig_name, rs, re)
        left_border = int(cs.LENGTH/2 - 2)
        right_border = int(cs.LENGTH/2 + 2)
        ref1 = np.concatenate(create_one_hot(ref))

        for x in range(0, len(ref)-cs.LENGTH, 5):
            start = bts[bs+x+left_border]
            end = bts[bs+x+right_border]
            number_of_signals = end - start + 1

            if number_of_signals < cs.SIGNAL_LENGTH:
                d = int((cs.SIGNAL_LENGTH - number_of_signals) / 2)
                signal_relevant_start = bs+x+left_border - d
                signal_relevant_end = bs+x + left_border + number_of_signals + d - 1 \
                    if number_of_signals + 2*d == cs.SIGNAL_LENGTH else \
                    bs + x + left_border + number_of_signals + d
            else:
                continue

            signal_relevant = []
            [signal_relevant.append(x) for x in norm_sig[signal_relevant_start:signal_relevant_end+1]]
            id_sig, std = ideal_signal_for_sequence(ref[x:x+cs.LENGTH])
            help_con = np.concatenate((ref1[4*x:4*(x+cs.LENGTH)], np.array(signal_relevant)), axis=0)
            help_con = np.concatenate((help_con, id_sig), axis=0)
            help_con = np.concatenate((help_con, [std]), axis=0)

            if len(help_con) != cs.NN_VECTOR_LENGTH:
                break
            vectors_for_nn = np.append(vectors_for_nn, help_con[None, :], axis=0)

    return vectors_for_nn


def create_true_vectors(filename, reference_file):
    """
    Vytvori cs.NUMBER_OF_EXAMLES pripadov klasifikovanych ako 1.
    """

    all_cases = np.array([], dtype=np.int64).reshape(0, cs.NN_VECTOR_LENGTH)
    alignment_file = open(filename, 'r')

    for line in alignment_file:
        line = line.split('\t')
        if len(line) >= 9:
            vector_nn = np.array(reference_to_signal_partial_mapping(line[9], reference_file, line[0][3:], line[1],
                                                                     int(line[2]), int(line[4])))
            all_cases = np.append(all_cases, vector_nn, axis=0)
            if len(all_cases) > cs.NUMBER_OF_EXAMPLES:
                break

    return all_cases


def save_vectors(np_array, save_file):
    np.save(save_file, np_array)


def signal_values(filename):
    """
    Vytvori globalny slovnik (k-mer, idealny signal).
    """

    global standard_deviation
    f = h5py.File(filename, 'r')
    grp = np.array(f.get('/model'))

    for i in grp:
        ideal_signal_values[str(i[0])[2:-1]] = {cs.VALUE: i[1]}
    standard_deviation = np.array(grp[0][2])
    return ideal_signal_values


def create_one_hot(sequence):
    """
    Vytvori one-hot reprezentaciu DNA.
    """

    transformed = []
    for x in sequence:
        transformed.append(cs.DNA[x])
    return np.array(transformed)


def ideal_signal_for_sequence(sequence):
    """
    Pre sekvenciu vytvori idealnu postupnost signalu podla 'kmer_model.hdf5'.
    """

    id_sig = []
    # standard_deviation = ideal_signal_values[0][cs.STD]

    for x in range(0, len(sequence) - cs.KMER_LENGTH + 1):
        id_sig.append(ideal_signal_values[str(''.join(sequence[x:x+cs.KMER_LENGTH]))][cs.VALUE])

    return np.array(id_sig), np.array(standard_deviation)


def generate_random_sequence():
    """
    Vygeneruje nahodnu postupnost baz
    """

    seq = []
    [seq.append(np.random.choice(cs.DNA_BASES)) for _ in range(cs.LENGTH)]

    return seq


def generate_false(read_location):
    """
    Vytvori vektory klasifikovane ako 0, na zaklade dat z 'read_location' (pouzije len hodnoty normalizovaneho signalu).
    """

    f = h5py.File(read_location, 'r')
    grp = np.array(f.get('/Analyses/Basecall_1D_000/BaseCalled_template/Events'))
    norm_sig = normalized_signal(grp)
    vectors_for_nn = np.array([], dtype=np.int64).reshape(0, cs.NN_VECTOR_LENGTH)

    for i in range(0, (len(norm_sig) // cs.LENGTH - 1) * cs.LENGTH, cs.LENGTH):
        current_signal = np.array(norm_sig[i:i+cs.SIGNAL_LENGTH])
        current_sequence = generate_random_sequence()
        one_hot_sequence = np.concatenate(create_one_hot(current_sequence))
        id_sig, std = ideal_signal_for_sequence(current_sequence)
        help_con = np.concatenate((one_hot_sequence, current_signal), axis=0)
        help_con = np.concatenate((help_con, id_sig), axis=0)
        help_con = np.append(help_con, [std], axis=0)
        # print(vectors_for_nn.shape, )
        vectors_for_nn = np.append(vectors_for_nn, help_con[None, :], axis=0)

    return vectors_for_nn


def create_false_vectors(directory):
    """
    Vytvori cs.NUMBER_OF_EXAMLES pripadov klasifikovanych ako 0.
    """

    all_cases = np.array([], dtype=np.int64).reshape(0, cs.NN_VECTOR_LENGTH)

    for file in glob.glob(directory + "/*.fast5"):
        vector_nn = generate_false(file)
        all_cases = np.append(all_cases, vector_nn, axis=0)
        if len(all_cases) > cs.NUMBER_OF_EXAMPLES:
            break

    return all_cases


def my_shuffle_set(pos, neg):
    all_cases = len(pos) + len(neg)
    random_shuffle = np.array([], dtype=np.int64).reshape(0, cs.NN_VECTOR_LENGTH)

    target_values = []
    index_pos = 0
    index_neg = 0

    for _ in range(all_cases):
        x = random.randint(0, 1)
        if x == 1:
            random_shuffle = np.append(random_shuffle, pos[index_pos][None, :], axis=0)
            index_pos += 1
            target_values.append(1)
        else:
            random_shuffle = np.append(random_shuffle, neg[index_neg][None, :], axis=0)
            index_neg += 1
            target_values.append(0)

        if len(pos) == index_pos or len(neg) == index_neg:
            break

    for i in range(index_pos, len(pos)):
        random_shuffle = np.append(random_shuffle, pos[i][None, :], axis=0)
        target_values.append(1)

    for i in range(index_neg, len(neg)):
        random_shuffle = np.append(random_shuffle, neg[i][None, :], axis=0)
        target_values.append(0)

    return random_shuffle, target_values


if __name__ == '__main__':

    signal_values("kmer_model.hdf5")
    reference = "sapSuaA1.fa"
    korman_output_file = "myfile.txt"
    read = "hdf5_files/magnu_20180917_FAK01752_MN25854_sequencing_run_sapSua_39950_read_5432_ch_101_strand.fast5"

    training_negative = create_false_vectors('hdf5_files')
    save_vectors(training_negative, 'neg')

    training_positive = create_true_vectors(korman_output_file, reference)
    save_vectors(training_positive, 'pos')

    print(np.load('pos.npy'))
    print(len(np.load('pos.npy')))

    shuffle, target = my_shuffle_set(np.load('pos.npy'), np.load('neg.npy'))

    save_vectors(shuffle, 'shuffle')
    save_vectors(target, 'target')
    print(np.load('shuffle.npy')[1])

    a = np.load('shuffle.npy')
    b = np.load('target.npy')

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(a, b, test_size=0.6, random_state=42)
    np.savetxt("shuffle.csv", X_train, delimiter=",")
    np.savetxt("target.csv", y_train, delimiter=",")

    X_train, X_test, y_train, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
    np.savetxt("test_X.csv", X_train, delimiter=",")
    np.savetxt("test_Y.csv", y_train, delimiter=",")
