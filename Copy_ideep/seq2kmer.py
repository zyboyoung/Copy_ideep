# 将读取的序列（一个文件中含有1000条，每条长度为101）的4-mer特征提取出来，生成特征矩阵（1000， 256）
# 对于四种碱基的组合，给出所有256种可能
import gzip
import numpy as np
from time import time


def get_4_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base = len(chars)
    end = len(chars) ** 4
    for i in range(0, end):
        n = i
        ch0 = chars[n % base]
        n = int(n / base)
        ch1 = chars[n % base]
        n = int(n / base)
        ch2 = chars[n % base]
        n = int(n / base)
        ch3 = chars[n % base]
        nucle_com.append(ch0 + ch1 + ch2 + ch3)

    return nucle_com


# 对于给定的序列，统计出每种4-kmer的频次，生成256维的特征向量
def get_4_nucleotide_composition(tris, seq, pythoncount=True):
    """
    :param tris: 
    :param seq: 
    :param pythoncount: 
    :return: 
    """
    seq_len = len(seq)

    # 将所有碱基大写，同时将部分T碱基替换为U碱基
    seq = seq.upper().replace('T', 'U')
    tri_feature = []

    if pythoncount:
        for val in tris:
            num = seq.count(val)
            tri_feature.append(num / seq_len)
    else:
        k = len(tris[0])
        tmp_fea = [0] * len(tris)
        for x in range(seq_len + 1 - k):
            kmer = seq[x: x + k]
            if kmer in tris:
                ind = tris.index(kmer)
                tmp_fea[ind] = tmp_fea[ind] + 1
        tri_feature = [val / seq_len for val in tmp_fea]

    return tri_feature


# 一共1000条序列，序列长度都为101，分别得到其4-mer组成，形成特征矩阵
def read_oli_feature(seq_file):
    trids4 = get_4_trids()
    seq_list = []
    seq = ''
    with gzip.open(seq_file, 'r') as fp:
        for line in fp:
            line = line.decode()
            if line[0] == '>':
                name = line[1: -1]
                if len(seq):
                    seq_array = get_4_nucleotide_composition(trids4, seq)
                    seq_list.append(seq_array)
                seq = ''
            else:
                seq = seq + line[: -1]

        if len(seq):
            seq_array = get_4_nucleotide_composition(trids4, seq)
            seq_list.append(seq_array)

    return np.array(seq_list)


if __name__ == '__main__':
    start_time = time()
    a = read_oli_feature('sequences.fa.gz')
    print(a)
    end_time = time()

    print('------ cost ' + str(end_time - start_time) + ' s ------')
