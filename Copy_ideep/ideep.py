import numpy as np
import os
from time import time
import argparse
from train_model import train_ideep
from predict_model import test_ideep

# np.set_printoptions(threshold=np.nan)


def calculate_performance(test_num, pred_y, labels):
    """
    :param test_num: 测试集数量
    :param pred_y: 预测的结果
    :param labels: 实际分类结果
    :return: 准确度acc、查准率precision、查全率/灵敏度sensitivity、特异性specificity、马修斯相关系数mcc
    """
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] == 1:
            if labels[index] == pred_y[index]:
                tp += 1
            else:
                fn += 1
        else:
            if labels[index] == pred_y[index]:
                tn += 1
            else:
                fp += 1

    # 分别求准确度acc、查准率precision、查全率/灵敏度sensitivity、特异性specificity、马修斯相关系数mcc
    acc = (tp + tn) / test_num
    precision = tp / (tp + fp)
    sensitivity = tp / (tp + fn)
    specitivity = tn / (tn + fp)
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return acc, precision, sensitivity, specitivity, mcc


def run_ideep(parser):
    data_dir = parser.data_dir
    out_file = parser.out_file
    train = parser.train
    model_dir = parser.model_dir
    predict = parser.predict
    seq = parser.seq
    region_type = parser.region_type
    cobinding = parser.cobinding
    structure = parser.structure
    motif = parser.motif
    batch_size = parser.batch_size
    n_epochs = parser.n_epochs

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if predict:
        train = False

    if train:
        print('model training')
        train_ideep(data_dir, model_dir, rg=region_type, clip=cobinding, rna=structure,
                    motif=motif, seq=seq, batch_size=batch_size, epoch=n_epochs)

    else:
        print('model prediction')
        test_ideep(data_dir, model_dir, outfile = out_file, rg=region_type,
                   clip=cobinding, rna=structure, motif = motif, seq = seq)


# 命令行解析
def parse_arguments(parser):
    parser.add_argument('--data_dir', type=str, metavar='<data_directory>',
                        help='Under this directory, you should have feature file: '
                             'sequences.fa.gz, matrix_Response.tab.gz, '
                             'matrix_RegionType.tab.gz, matrix_RNAfold.tab.gzmatrix_Cobinding.tab.gz, '
                             'motif_fea.gz, and label file matrix_Response.tab.gz with 0 and 1 ')
    parser.add_argument('--train', type=bool, default=True, help='use this option for training model')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='The directory to save the trained models for future prediction')
    parser.add_argument('--predict', type=bool, default=False,
                        help='Predicting the RNA-protein binding sites for your input sequences, '
                             'if using train, then it will be False')
    parser.add_argument('--out_file', type=str, default='prediction.txt',
                        help='The output file used to store the prediction probability of testing data')
    parser.add_argument('--seq', type=bool, default=True, help='The sequences feature for Convolutional neural network')
    parser.add_argument('--region_type', type=bool, default=True,
                        help='The modularity of region type (types (exon, intron, 5UTR, 3UTR, CDS)')
    parser.add_argument('--cobinding', type=bool, default=True, help='The modularity of cobinding')
    parser.add_argument('--structure', type=bool, default=True,
                        help='The modularity of structure that is probability of RNA secondary structure')
    parser.add_argument('--motif', type=bool, default=False, help='The modularity of motif scores')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='The size of a single mini-batch (default value: 100)')
    parser.add_argument('--n_epochs', type=int, default=20, help='The number of training epochs (default value: 20)')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    start_time = time()

    parser = parse_arguments(argparse.ArgumentParser())
    run_ideep(parser)

    end_time = time()

    print('------ cost ' + str(end_time - start_time) + ' s ------')
