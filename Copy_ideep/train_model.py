from preprocess import load_data, preprocess_data
import numpy as np
import random
from sklearn.externals import joblib
from cnn_network import get_rnn_fea, get_cnn_network
from preprocess import preprocess_labels
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Concatenate


# 将原始数据划分为训练集和验证集，比例为8:2
def split_training_validation(classes, validation_size=0.2, shuffle=False):
	# 样本总数: 5000
	num_samples = len(classes)

	# 种类的数量: 2
	classes_unique = np.unique(classes)

	indices = np.arange(num_samples)
	training_indice = []
	training_label = []
	validation_indice = []
	validation_label = []

	for cl in classes_unique:
		# 找到各类样本的index
		indices_cl = indices[classes == cl]
		# 各类样本的数量： 4000（class:0）： 1000（class:1）
		num_samples_cl = len(indices_cl)

		if shuffle:
			random.shuffle(indices_cl)

		# 按比例（0.2）取验证集
		num_samples_each_split = int(num_samples_cl * validation_size)
		res = num_samples_cl - num_samples_each_split

		training_indice += [val for val in indices_cl[num_samples_each_split:]]
		training_label = training_label + [cl] * res

		validation_indice += [val for val in indices_cl[num_samples_each_split:]]
		validation_label = validation_label + [cl] * num_samples_each_split

	# 处理后，训练集一共4000样本，其中3200个0类集，800个1类集，验证集一共1000样本，其中800个0类集，200个1类集
	# 分别将训练集和验证集中的样本顺序打乱，标签按同样的方法打乱
	training_index = np.arange(len(training_indice))
	random.shuffle(training_index)
	training_indice = np.array(training_indice)[training_index]
	training_label = np.array(training_label)[training_index]

	validation_index = np.arange(len(validation_label))
	random.shuffle(validation_index)
	validation_indice = np.array(validation_indice)[validation_index]
	validation_label = np.array(validation_label)[validation_index]

	return training_indice, training_label, validation_indice, validation_label


def train_ideep(data_dir, model_dir, rg=True, clip=True, rna=True, motif=False, seq=True, batch_size=100, epoch=20):

	training_data = load_data(data_dir, rg=rg, clip=clip, rna=rna, motif=motif, seq=seq)
	print('training', len(training_data))

	rg_hid = 128
	clip_hid = 256
	rna_hid = 64
	cnn_hid = 64
	motif_hid = 64
	seq_hid = 102

	# 将样本8:2分为训练集和验证集，并且打乱顺序
	training_indice, training_label, validation_indice, validation_label = split_training_validation(training_data['Y'])

	if rg:
		rg_data, rg_scaler = preprocess_data(training_data['X_RG'])

		# 将训练的模型存到本地
		joblib.dump(rg_scaler, os.path.join(model_dir, 'rg_scaler.pkl'))

		# 从原数据中根据index取出训练集
		rg_train = rg_data[training_indice]

		# 从原数据中根据index取出验证集
		rg_validation = rg_data[validation_indice]

		rg_net = get_rnn_fea(rg_train, sec_num_hidden=rg_hid, num_hidden=rg_hid * 2)
		rg_data = []
		training_data["X_RG"] = []

	if clip:
		clip_data, clip_scaler = preprocess_data(training_data["X_CLIP"])
		joblib.dump(clip_scaler, os.path.join(model_dir, 'clip_scaler.pkl'))
		clip_train = clip_data[training_indice]
		clip_validation = clip_data[validation_indice]
		clip_net = get_rnn_fea(clip_train, sec_num_hidden=clip_hid, num_hidden=clip_hid * 3)
		clip_data = []
		training_data["X_CLIP"] = []

	if rna:
		rna_data, rna_scaler = preprocess_data(training_data["X_RNA"], stand=True)
		joblib.dump(rna_scaler, os.path.join(model_dir, 'rna_scaler.pkl'))
		rna_train = rna_data[training_indice]
		rna_validation = rna_data[validation_indice]
		rna_net = get_rnn_fea(rna_train, sec_num_hidden=rna_hid, num_hidden=rna_hid * 2)
		rna_data = []
		training_data["X_RNA"] = []

	if motif:
		motif_data, motif_scaler = preprocess_data(training_data["motif"], stand=True)
		joblib.dump(motif_scaler, os.path.join(model_dir, 'motif_scaler.pkl'))
		motif_train = motif_data[training_indice]
		motif_validation = motif_data[validation_indice]
		motif_net = get_rnn_fea(motif_train, sec_num_hidden=motif_hid, num_hidden=motif_hid * 2)  # get_cnn_network()
		motif_data = []
		training_data["motif"] = []

	if seq:
		seq_data = training_data["seq"]
		seq_train = seq_data[training_indice]
		seq_validation = seq_data[validation_indice]
		seq_net = get_cnn_network()
		seq_data = []

	y, encoder = preprocess_labels(training_label)
	val_y, encoder = preprocess_labels(validation_label, encoder=encoder)
	training_data.clear()

	training_net = []
	training = []
	validation = []
	total_hid = 0

	if rg:
		training_net.append(rg_net)
		training.append(rg_train)
		validation.append(rg_validation)
		total_hid = total_hid + rg_hid
		rg_train = []
		rg_validation = []

	if clip:
		training_net.append(clip_net)
		training.append(clip_train)
		validation.append(clip_validation)
		total_hid = total_hid + clip_hid
		clip_train = []
		clip_validation = []

	if rna:
		training_net.append(rna_net)
		training.append(rna_train)
		validation.append(rna_validation)
		total_hid = total_hid + rna_hid
		rna_train = []
		rna_validation = []

	if motif:
		training_net.append(motif_net)
		training.append(motif_train)
		validation.append(motif_validation)
		total_hid = total_hid + motif_hid
		motif_train = []
		motif_validation = []

	if seq:
		training_net.append(seq_net)
		training.append(seq_train)
		validation.append(seq_validation)
		total_hid = total_hid + seq_hid
		seq_train = []
		seq_validation = []

	model = Sequential()
	model.add()


if __name__ == '__main__':
	data = load_data('./datasets/clip/1_PARCLIP_AGO1234_hg19/5000/training_sample_0/')
	# training_indice, training_label, validation_indice, validation_label = split_training_validation(data['Y'])
	# rg_data, rg_scaler = preprocess_data(data['X_RG'])
	# rg_train = rg_data[training_indice]


