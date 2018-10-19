"""
read RNA seq and transfer to concolutional array
"""

import numpy as np
import gzip
import pdb
from time import time

np.set_printoptions(threshold=np.nan)


# 从压缩文件中读取出序列，得到卷积用的array(1000, 107, 4)
def read_seq(seq_file):
    """
    :param seq_file: 
    :return: seq_list
    """
    seq_list = []
    seq = ''
    with gzip.open(seq_file, 'r') as fp:
        for line in fp:
            line = line.decode()
            if line[0] == '>':
                name = line[1: -1]
                if len(seq):
                    seq_array = get_RNA_seq_convolutional_array(seq)
                    seq_list.append(seq_array)
                seq = ''
            else:
                seq = seq + line[: -1]
        if len(seq):
            seq_array = get_RNA_seq_convolutional_array(seq)
            seq_list.append(seq_array)

    return np.array(seq_list)


# 给序列加上padding，生成卷积array
def get_RNA_seq_convolutional_array(seq, motif_len=4):
    """
    :param seq: 
    :param motif_len: 
    :return: 
    """
    seq = seq.replace('U', 'T')
    alpha = 'ACGT'
    row = len(seq) + 2 * motif_len - 2
    new_array = np.zeros((row, 4))

    for i in range(motif_len - 1):
        new_array[i] = np.array([0.25] * 4)

    for i in range(row - 3, row):
        new_array[i] = np.array([0.25] * 4)

    for i, val in enumerate(seq):
        i = i + motif_len - 1
        if val not in 'ACGT':
            new_array[i] = np.array([0.25] * 4)
            continue
        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except:
            pdb.set_trace()

    return new_array


if __name__ == '__main__':
    start_time = time()
    print(read_seq('sequences.fa.gz'))
    end_time = time()

    print('------ cost ' + str(end_time - start_time) + ' s ------')
