import numpy as np
import gzip
import os
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from seq2array import read_seq
from seq2kmer import read_oli_feature


# 根据序列得到其互补配对的序列
def complement(seq):
    complement_method = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    complseq = [complement_method[base] for base in seq]

    return complseq


# 根据序列得到其反向互补序列
def reverse_complement(seq):
    seq = list(seq)
    seq.reverse()
    return ''.join(complement(seq))


# 从多源数据中获得标签
def load_labels(path, kmer=True, rg=True, clip=True, rna=True, go=True):
    """
    :param path: 
    :param kmer: 
    :param rg: 
    :param clip: 
    :param rna: 
    :param go: 
    :return: labels-dict
    """
    labels = dict()
    if go:
        labels['X_GO'] = gzip.open(os.path.join(path, 'matrix_GeneOntology.tab.gz')).readline().decode().split('\t')
    if kmer:
        labels['X_KMER'] = gzip.open(os.path.join(path, 'matrix_RNAkmers.tab.gz')).readline().decode().split('\t')
    if rg:
        labels['X_RG'] = gzip.open(os.path.join(path, 'matrix_RegionType.tab.gz')).readline().decode().split('\t')
    if clip:
        labels['X_CLIP'] = gzip.open(os.path.join(path, 'matrix_Cobinding.tab.gz')).readline().decode().split('\t')
    if rna:
        labels['X_RNA'] = gzip.open(os.path.join(path, 'matrix_RNAfold.tab.gz')).readline().decode().split('\t')

    return labels


# 将标签转化为标准的LabelEncoder对象，并且通过one-hot编码转化为特征矩阵
def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        # 获取标签值
        encoder = LabelEncoder()
        encoder.fit(labels)
    # 标签值进行标准化
    y = encoder.transform(labels).astype(np.int32)

    if categorical:
        # 将标签值通过one-hot编码转化为矩阵
        y = np_utils.to_categorical(y)

    return y, encoder


# 使用字典存储多源数据
def load_data(path, kmer=False, rg=True, clip=True, rna=True, go=False, motif=True, seq=True, oli=False, test=False):
    data = dict()
    if go:
        data['X_GO'] = np.loadtxt(gzip.open(os.path.join(path, 'matrix_GeneOntology.tab.gz')), skiprows=1)
    if kmer:
        data['X_KMER'] = np.loadtxt(gzip.open(os.path.join(path, 'matrix_RNAkmers.tab.gz')), skiprows=1)
    if rg:
        data['X_RG'] = np.loadtxt(gzip.open(os.path.join(path, 'matrix_RegionType.tab.gz')), skiprows=1)
    if clip:
        data['X_CLIP'] = np.loadtxt(gzip.open(os.path.join(path, 'matrix_Cobinding.tab.gz')), skiprows=1)
    if rna:
        data['X_RNA'] = np.loadtxt(gzip.open(os.path.join(path, 'matrix_RNAfold.tab.gz')), skiprows=1)
    if motif:
        data['motif'] = np.loadtxt(gzip.open(os.path.join(path, 'motif_fea.gz')), skiprows=1, usecols=range(1, 103))
    if seq:
        data['seq'] = read_seq(os.path.join(path, 'sequences.fa.gz'))
    if oli:
        data['oli'] = read_oli_feature(os.path.join(path, 'sequences.fa.gz'))
    if test:
        data['Y'] = []
    else:
        data['Y'] = np.loadtxt(gzip.open(os.path.join(path, 'matrix_Response.tab.gz')), skiprows=1)

    return data


# 可以选择对原数据进行标准化（均值为0，方差为1），或者归一化（最小值设置为0，最大值设置为1）
def preprocess_data(X, scaler=None, stand=False):
    if not scaler:
        if stand:
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        scaler.fit(X)
    X = scaler.transform(X)

    return X, scaler


# 22条常染色体，1条X染色体，1条Y染色体，1条线粒体基因组，返回的是字典类型sequences['chr1'] = sequence
def get_hg19_sequence():
    chr_focus = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9',
                 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17',
                 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX', 'chrY', 'chrM']

    sequences = {}
    dir1 = 'hg19_seq/'
    for chr_name in chr_focus:
        file_name = chr_name + '.fa.gz'
        if not os.path.exists(dir1 + file_name):
            print('download ' + chr_name + ' genome sequence file')
            cli_str = 'rsync -avzP rsync://hgdownload.cse.ucsc.edu/goldenPath/hg19/chromosomes/' \
                      + chr_name + '.fa.gz ' + dir1
            fex = os.popen(cli_str, 'r')
            fex.close()

        print('file %s' % file_name)

        with gzip.open(dir1 + file_name, 'r') as fp:
            sequence = ''
            for line in fp:
                line = line.decode()
                if line[0] == '>':
                    name = line.split()[0]
                else:
                    sequence = sequence + line.split()[0]
            sequences[chr_name] = sequence

    return sequences


if __name__ == '__main__':
    data = load_data('./datasets/clip/1_PARCLIP_AGO1234_hg19/5000/training_sample_0/')
    print(preprocess_data(data['X_RG']))
