from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from seq2array import get_RNA_seq_convolutional_array
import numpy as np
from preprocess import preprocess_labels
from time import time


# 卷积→激活→卷积→激活→最大池化→随机失活→展平
def get_2d_cnn_network():
	nb_conv = 4
	nb_pool = 2
	model = Sequential()
	model.add(Conv2D(64, kernel_size=nb_conv, padding='valid', input_shape=(1, 107, 4), data_format="channels_first"))
	model.add(Activation('relu'))
	model.add(Conv2D(64, kernel_size=nb_conv, data_format="channels_first"))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(nb_pool, 2))
	model.add(Dropout(0.25))
	model.add(Flatten())

	return model


def get_cnn_network():
	# 指定卷积核的数目，即输出的维度
	nbfilter = 102

	# 构建模型，向其中添加网络层
	model = Sequential()

	# 卷积层，指定卷积核的数量（输出维度）filters，卷积核的大小（kernel_size）
	model.add(Conv1D(input_dims=4, input_length=107, filters=nbfilter,
					 kernel_size=7, padding='valid', activation='relu'))

	model.add(MaxPooling1D(pool_length=3))

	# dropout层，以0.5为概率随机失活
	model.add(Dropout(0.5))

	# 将输入压平为一维，实现从卷积层到全连接层的过渡
	model.add(Flatten())

	# 添加全连接层，并指定激活函数
	model.add(Dense(nbfilter, activation='relu'))

	model.add(Dropout(0.25))

	return model


def run_cnn():
	X_train = []

	# seqs: [100, 101]
	seqs = 50 * ['CGUACACGGUGGAUGCCCUGGCAGUCAAGGCGAUGAAGGACGUGCUAAUCU'
				 'GCGAUAAGCGUCGGUAAGGUGAUAUGAACCGUUUAACCGGCGAUUUCCGC',
				 'GGCGGCCGUAGCGCGGUGGUCCCACCUGACCCCAUGCCGAACUCAGAAGUG'
				 'AAACGCCGUAGCGCCGAUGGUAGUGUGGGGUCUCCCCAUGCGAGAGUAGG']

	for seq in seqs:
		# 添加padding，one-hot编码，[107, 4]
		tmp_train = get_RNA_seq_convolutional_array(seq)
		# X_train: list，长度为100，每个元素是[107, 4]的array
		X_train.append(tmp_train)

	model = get_cnn_network()

	# y_train: 一维数组（100, ）
	y_train = np.array([0, 1] * 50)

	# 将y_train转化为标签矩阵，[100, 2]
	y_train, encoder = preprocess_labels(y_train)

	model.add(Dense(input_dim=64, units=2))
	model.add(Activation('sigmoid'))
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

	# np.array(X_train): [100, 107, 4]
	model.fit(np.array(X_train), y_train, batch_size=50, epochs=50)


def get_rnn_fea(train, sec_num_hidden=128, num_hidden=128):
	print('configure network for ', train.shape)
	model = Sequential()
	model.add(Dense(units=num_hidden, input_shape=(train.shape[1],), activation='relu'))
	model.add(PReLU())
	model.add(BatchNormalization(mode=2))
	model.add(Dropout(0.5))
	model.add(Dense(units=num_hidden, activation='relu'))
	model.add(PReLU())
	model.add(BatchNormalization(mode=2))
	model.add(Dropout(0.5))

	return model





if __name__ == '__main__':
	start_time = time()
	run_cnn()
	end_time = time()

	print('------ cost ' + str(end_time - start_time) + ' s ------')
